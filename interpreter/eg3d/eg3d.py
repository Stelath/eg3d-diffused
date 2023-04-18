import os
import io
import sys
import pickle
import numpy as np

import torch

from .camera_utils import LookAtPoseSampler, FOV_to_intrinsics

from diffuser_utils.utils import get_device

class EG3D():
    def __init__(self, model_path, device='cpu', render_only=False):
        self.device = torch.device(device)
        self.load_eg3d(model_path)
        self.render_only = render_only
        if render_only:
            self.G.decoder = self.G.decoder.to(self.device)
        else:
            self.G = self.G.to(self.device)
            self.generate_imgs(torch.zeros((1, 512)))
            
        
    def generate_imgs(self, latent_vector, transpose=False):
        assert not self.render_only
        torch.cuda.empty_cache()
        G = self.G
        latent_vector = latent_vector.to(self.device)
        cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=self.device), radius=2.7, device=self.device)
        intrinsics = FOV_to_intrinsics(18, device=self.device)
        
        cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=self.device)
        cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
        cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=self.device)
        conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=self.device)
        camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        camera_params = camera_params.repeat(latent_vector.shape[0], 1)
        
        conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        conditioning_params = conditioning_params.repeat(latent_vector.shape[0], 1)
        
        mapped_vectors = G.mapping(latent_vector, conditioning_params, truncation_psi=0.5, truncation_cutoff=8)
        
        out = self.G.synthesis(mapped_vectors, camera_params)
        imgs = out['image'].detach().cpu().numpy()
        
        for key in list(out.keys()):
            del out[key]
        del out
        torch.cuda.empty_cache()
        
        if transpose:
            imgs = imgs.transpose(0, 2, 3, 1)
            
        imgs = (np.clip(imgs, -1, 1) + 1) / 2

        return torch.from_numpy(imgs).to(self.device)
    
    def generate_random_img(self):
        latent_vector = torch.randn([1, self.G.z_dim]).to(self.device)
        return self.generate_imgs(latent_vector), latent_vector
    
    def generate_planes(self, latent_vector, reshape=True):
        assert not self.render_only
        G = self.G
        latent_vector = latent_vector.to(self.device)
        cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=self.device), radius=2.7, device=self.device)
        intrinsics = FOV_to_intrinsics(18, device=self.device)
        
        cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=self.device)
        cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
        cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=self.device)
        conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=self.device)
        camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        camera_params = camera_params.repeat(latent_vector.shape[0], 1)
        
        conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        conditioning_params = conditioning_params.repeat(latent_vector.shape[0], 1)
        
        mapped_vectors = G.mapping(latent_vector, conditioning_params, truncation_psi=0.5, truncation_cutoff=8)
        planes = self.synthesize_planes(mapped_vectors)
        
        if reshape:
            # Reshape output into three 32-channel planes
            planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        
        return planes
        
    def synthesize_planes(self, ws, update_emas=False, **synthesis_kwargs):
        # Create triplanes by running StyleGAN backbone
        planes = self.G.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        
        return planes
    
    def generate_random_planes(self, reshape=True):
        latent_vector = torch.randn([1, self.G.z_dim]).to(self.device)
        return self.generate_planes(latent_vector, reshape=reshape), latent_vector
    
    def render_planes(self, planes, camera_params, neural_rendering_resolution=None, transpose=True, reshape_planes=False):
        # TODO: IMPLEMENT CAMERA PARAMS FOR USER CONTROLL
        cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=self.device), radius=2.7, device=self.device)
        intrinsics = FOV_to_intrinsics(18, device=self.device)
        
        cam_pivot = torch.tensor(self.G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=self.device)
        cam_radius = self.G.rendering_kwargs.get('avg_camera_radius', 2.7)
        cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=self.device)
        
        cam2world_matrix = cam2world_pose.view(-1, 4, 4).repeat(planes.shape[0], 1, 1)
        intrinsics = intrinsics.view(-1, 3, 3).repeat(planes.shape[0], 1, 1)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.G.neural_rendering_resolution
        else:
            self.G.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.G.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)
        N, M, _ = ray_origins.shape
        
        if reshape_planes:
            planes = planes.view(planes.shape[0], 3, 32, planes.shape[-2], planes.shape[-1])
        
        # Perform volume rendering
        feature_samples, depth_samples, weights_samples = self.G.renderer(planes, self.G.decoder, ray_origins, ray_directions, self.G.rendering_kwargs) # channels last

        # Reshape into 'raw' neural-rendered image
        H = W = self.G.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]
        # sr_image = self.G.superresolution(rgb_image, feature_image, ws, noise_mode=self.G.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
        
        if transpose:
            rgb_image = rgb_image.permute(0, 2, 3, 1)
        
        return rgb_image
        
    def load_eg3d(self, model_path):
        with open(model_path, 'rb') as f:
            self.G = CPU_Unpickler(f).load()['G_ema']  # torch.nn.Module
    
    def update_device(self):
        self.device = get_device(self.G)


# Pulled From: https://stackoverflow.com/questions/57081727/load-pickle-file-obtained-from-gpu-to-cpu
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

