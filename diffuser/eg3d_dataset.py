import os
import numpy as np
import pandas as pd
from skimage import io
from diffuser_utils.encoding import create_attention_matrix

import torch
from torch.utils.data import Dataset

class EG3DDataset(Dataset):
    def __init__(self, df_file, data_dir, transform=None):
        self.eg3d_data = pd.read_pickle(os.path.join(data_dir, df_file))
        self.data_dir = data_dir
        self.transform = transform
        self.size = 512
        self.attention_matrix = create_attention_matrix(512, 512)
    
    def encode_latent_vector(self, vector):
        vector = np.tile(np.reshape(vector, (self.size, 1)), self.size)
        encoded_vector = self.attention_matrix + vector
        
        return encoded_vector
     
    def __len__(self):
        return len(self.eg3d_data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_path = os.path.join(self.data_dir, self.eg3d_data.iloc[idx, 0])
        image = io.imread(img_path)
        latent_vector = self.eg3d_data.iloc[idx, 1]
        item = {'image': image, 'encoded_vector': encoded_vector}
        
        if self.transform:
            item = self.transform(item)
            
        return item