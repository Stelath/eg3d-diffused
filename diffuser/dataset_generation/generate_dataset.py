import os
import sys
import pickle

import torch
import numpy as np
import pandas as pd
from skimage import io
from tqdm import tqdm

from camera_utils import LookAtPoseSampler, FOV_to_intrinsics

from ..utils import positionalEncoding

def generate_img(G, device='cuda'):
    cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
    intrinsics = FOV_to_intrinsics(18, device=device)

    c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1) # camera parameters
    latent_vector = torch.randn([1, G.z_dim]).cuda()    # latent codes

    # img = G(z, c)['image']                           # NCHW, float32, dynamic range [-1, +1], no truncation

    cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
    cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
    cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
    conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
    camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    w = G.mapping(latent_vector, conditioning_params, truncation_psi=0.5, truncation_cutoff=8)
    img = G.synthesis(w, camera_params)['image']
    return img, latent_vector

def encode_latent_vector(vector):
    size = vector.shape[0]
    product = positionalEncoding(size, size) * vector
    
    return product

def create_img_encoded_pair(G, device='cuda'):
    img, latent_vector = generate_img(G, device=device)
    encoded_vector = encode_latent_vector(latent_vector)
    
    return img, encoded_vector

def create_dataset(num_samples, out='data/', device='cuda'):
    with open('eg3d_model/ffhqrebalanced512-128.pkl', 'rb') as f:
        G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
    
    dataset = pd.DataFrame(columns=['image', 'encoded_vector'])
    for i in tqdm(range(num_samples)):
        img, encoded_vector = create_img_encoded_pair(G, device=device)
        file_name = f'imgs/{str(i+1).zfill(len(str(num_samples)))}.png'
        io.imsave(os.join(out, file_name), img)
        pd.concat([dataset, {'image': file_name, 'encoded_vector': encoded_vector}], axis=0)
    
    dataset.to_csv(os.join(out, 'dataset.csv'))
