import os
import sys
import pickle

import torch
import numpy as np
import pandas as pd
from skimage import io
from skimage import img_as_ubyte
from skimage.color import rgb2gray
from tqdm import tqdm

from camera_utils import LookAtPoseSampler, FOV_to_intrinsics

from encoding import positionalEncoding
from generate_dataset_args import parse_args

def generate_img(G, device='cuda'):
    cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
    intrinsics = FOV_to_intrinsics(18, device=device)
    latent_vector = torch.randn([1, G.z_dim]).to(device)

    cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
    cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
    cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
    conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
    camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    conditioning_params = conditioning_params.repeat(1, 1)
    mapped_vectors = G.mapping(latent_vector, conditioning_params, truncation_psi=0.5, truncation_cutoff=8)
    
    img = G.synthesis(mapped_vectors, camera_params)['image'][0]
    img = img.cpu().numpy().transpose(1, 2, 0)
    img = (np.clip(img, -1, 1) + 1) / 2
    
    return img, latent_vector.detach().cpu().numpy()[0]

# def encode_latent_vector(vector):
#     size = vector.shape[0]
#     vector = np.tile(np.reshape(vector, (size, 1)), size)
#     encoded_vector = positionalEncoding(size, size) + vector
    
#     return encoded_vector

# def create_img_encoded_pair(G, device='cuda'):
#     imgs, latent_vector = generate_img(G, device=device)
#     encoded_vector = encode_latent_vector(latent_vector)
    
#     return imgs, latent_vector, encoded_vector

def create_dataset(model_path, num_samples, out='data/', device='cuda'):
    np.set_printoptions(threshold=sys.maxsize)
    with open(model_path, 'rb') as f:
        G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
    
    dataset = pd.DataFrame(columns=['image', 'latent_vector'])
    for i in tqdm(range(num_samples)):
        img, latent_vector = generate_img(G, device=device)
        
        file_name = f'imgs/{str(i+1).zfill(len(str(num_samples)))}.png'
        img = rgb2gray(img)
        img = img_as_ubyte(img)
        io.imsave(os.path.join(out, file_name), img)
        
        dataset.loc[len(dataset.index)] = [file_name, list(latent_vector)]
    
    dataset.to_csv(os.path.join(out, 'dataset.csv'))

if __name__ == "__main__":
    args = parse_args()
    create_dataset(args.model_path, args.num_samples, out=args.out_dir)
