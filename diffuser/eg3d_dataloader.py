import pandas as pd
from skimage import io
import os

import torch
from torch.utils.data import Dataset

class EG3DDataDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.eg3d_data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.eg3d_data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = os.path.join(self.root_dir, self.eg3d_data.iloc[idx, 0])
        image = io.imread(img_name)
        label = self.eg3d_data.iloc[idx, 1]
        sample = {'image': image, 'label': label}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample