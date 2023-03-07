import os
import sys
import pickle
import cv2

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from deepface import DeepFace

import copy
from SPIGA.inference.config import ModelConfig
from SPIGA.inference.framework import SPIGAFramework

from diffuser_utils.augment_dataset_args import parse_args

def augment_dataset(data_dir, device='cuda'):
    dataset = 'wflw'
    model_cfg = ModelConfig(dataset)
    processor = SPIGAFramework(model_cfg)
    
    dataset = pd.read_pickle(os.path.join(data_dir, "dataset.df"))
    analysis = []
    landmarks = []
    encoding = []
    
    for i, row in tqdm(dataset.iterrows()):
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

if __name__ == "__main__":
    args = parse_args()
    augment_dataset(args.dataset)
