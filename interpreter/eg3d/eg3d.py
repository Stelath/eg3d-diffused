import os
import sys
import pickle
import numpy as np

import torch

from .camera_utils import LookAtPoseSampler, FOV_to_intrinsics

class EG3D():
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.load_eg3d(model_path)
        
    def generate_imgs(self, latent_vector):
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
        
        # imgs = imgs.transpose(0, 2, 3, 1)
        imgs = (np.clip(imgs, -1, 1) + 1) / 2

        return torch.from_numpy(imgs).to(self.device)

    def generate_random_img(self):
        latent_vector = torch.randn([1, self.G.z_dim]).to(self.device)
        return self.generate_imgs(latent_vector), latent_vector.cpu().numpy()[0]
    
    def load_eg3d(self, model_path):
        with open(model_path, 'rb') as f:
            self.G = pickle.load(f)['G_ema'].to(self.device)  # torch.nn.Module