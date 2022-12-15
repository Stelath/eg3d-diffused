import os
import numpy as np
import pandas as pd
from skimage import io
from skimage import img_as_float32
from diffuser_utils.encoding import create_attention_matrix

import torch
from torch.utils.data import Dataset

class EG3DDataset(Dataset):
    def __init__(self, df_file, data_dir, transform=None, vector_size=512):
        self.eg3d_data = pd.read_pickle(os.path.join(data_dir, df_file))
        self.data_dir = data_dir
        self.transform = transform
        self.size = vector_size
        self.attention_matrix = create_attention_matrix(vector_size, vector_size)
    
    def encode_latent_vector(self, vector):
        vector_matrix = np.tile(np.reshape(vector, (self.size, 1)), self.size)
        encoded_vector = self.attention_matrix + vector_matrix
        
        return encoded_vector
     
    def __len__(self):
        return len(self.eg3d_data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_path = os.path.join(self.data_dir, self.eg3d_data.iloc[idx, 0])
        image = io.imread(img_path)
        image = (img_as_float32(image)*2) - 1
        image = np.array(image, dtype=np.float16)
        
        latent_vector = self.eg3d_data.iloc[idx, 1]
        encoded_vector = self.encode_latent_vector(latent_vector)
        item = {'images': image, 'encoded_vectors': encoded_vector.astype(np.float16)}
        
        if self.transform:
            item = self.transform(item)
            
        return item