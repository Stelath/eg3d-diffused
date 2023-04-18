import os
import sys
import pickle
import cv2

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from deepface import DeepFace

import pickle

import copy
# from SPIGA.inference.config import ModelConfig
# from SPIGA.inference.framework import SPIGAFramework

from autoencoder import AutoencoderKLConfig, AutoencoderKL

from diffuser_utils.augment_dataset_args import parse_args

def add_faces(data_dir, device='cpu'):
    dataset = 'wflw'
    model_cfg = ModelConfig(dataset)
    processor = SPIGAFramework(model_cfg)
    
    dataset = pd.read_pickle(os.path.join(data_dir, "dataset.df"))
    analysis = []
    landmarks = []
    encoding = []
    
    for i, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
        image_pth = os.path.join(data_dir, row['image'])
        embedding_objs = DeepFace.represent(img_path = image_pth, model_name = "Facenet512", enforce_detection=False)
        encoding.append(embedding_objs[0]['embedding'])
        
        objs = DeepFace.analyze(img_path = image_pth, actions = ['age', 'gender', 'race', 'emotion'], silent=True, enforce_detection=False)
        analysis.append(objs[0])
        
        image = cv2.imread(image_pth)
        bbox = [0, 0, 512, 512]
        
        features = processor.inference(image, [bbox])
        landmark = np.array(features['landmarks'][0])
        landmarks.append(landmark)
        
    dataset['analysis'] = analysis
    dataset['landmarks'] = landmarks
    dataset['encoding'] = encoding
    dataset.to_pickle(os.path.join(data_dir, 'augmented_dataset.df'))


@torch.no_grad()
def add_ae_latent_encodings(data_dir, model_path, device='cpu'):
    os.makedirs(os.path.join(data_dir, 'encoded_triplanes'), exist_ok=True)
    
    ae = AutoencoderKL.load_from_checkpoint(model_path, strict=False)
    ae = ae.to(device)
    ae.eval()
    
    dataset = pd.read_pickle(os.path.join(data_dir, "dataset.df"))
    triplanes = np.memmap(os.path.join(data_dir, 'triplanes.mmap'), dtype='float32', mode='r', shape=(len(dataset.index), 96, 256, 256))
    
    for i, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
        inp = torch.from_numpy(triplanes[row.triplane_idx]).unsqueeze(0).to(device)
        out = ae.encode(inp)
        
        with open(os.path.join(data_dir, 'encoded_triplanes', f'{row.triplane_idx:04d}.pkl'), 'wb') as file:
            pickle.dump(out, file)
        

if __name__ == "__main__":
    args = parse_args()
    device = 'cuda' if args.gpu else 'cpu'
    if args.faces:
        add_faces(args.dataset, device)
    if args.ae:
        add_ae_latent_encodings(args.dataset, args.model_path, device)
    
