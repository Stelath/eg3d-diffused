import pandas as pd
from skimage import io
import os

import torch
from torch.utils.data import Dataset

class EG3DDataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None):
        self.eg3d_data = pd.read_csv(os.path.join(data_dir, csv_file))
        self.data_dir = data_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.eg3d_data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_path = os.path.join(self.data_dir, self.eg3d_data.iloc[idx, 0])
        image = io.imread(img_path)
        encoded_vector = self.eg3d_data.iloc[idx, 1]
        item = {'image': image, 'encoded_vector': encoded_vector}
        
        if self.transform:
            item = self.transform(item)
            
        return item