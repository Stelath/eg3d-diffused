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

from diffuser_utils.generate_dataset_args import parse_args



# def encode_latent_vector(vector):
#     size = vector.shape[0]
#     vector = np.tile(np.reshape(vector, (size, 1)), size)
#     encoded_vector = positionalEncoding(size, size) + vector
    
#     return encoded_vector

# def create_img_encoded_pair(G, device='cuda'):
#     imgs, latent_vector = generate_img(G, device=device)
#     encoded_vector = encode_latent_vector(latent_vector)
    
#     return imgs, latent_vector, encoded_vector

def create_dataset(model_path, num_samples, grayscale=False, out='data/', device='cuda'):
    eg3d = EG3D(model_path)
    
    dataset = pd.DataFrame(columns=['image', 'latent_vector'])
    for i in tqdm(range(num_samples)):
        img, latent_vector = eg3d.generate_random_img(device=device)
        
        file_name = f'imgs/{str(i+1).zfill(len(str(num_samples)))}.png'
        if grayscale:
            img = rgb2gray(img)
        img = img_as_ubyte(img)
        io.imsave(os.path.join(out, file_name), img)
        
        dataset.loc[len(dataset.index)] = [file_name, latent_vector]
    
    dataset.to_pickle(os.path.join(out, 'dataset.df'))

if __name__ == "__main__":
    args = parse_args()
    create_dataset(args.model_path, args.num_samples, grayscale=args.grayscale, out=args.out_dir)
