import os
import numpy as np
import pandas as pd
from PIL import Image
from diffuser_utils.encoding import create_attention_matrix

import torch
from torch.utils.data import Dataset

from transformers import BlipProcessor, CLIPImageProcessor

class EG3DDataset(Dataset):
    def __init__(self, df_file, data_dir, image_size=512, transform=None, encode=True):
        self.eg3d_data = pd.read_pickle(os.path.join(data_dir, df_file))
        self.data_dir = data_dir
        self.transform = transform
        self.size = image_size
        self.attention_matrix = create_attention_matrix(self.size, self.size)
        self.encode = encode
    
    def encode_latent_vector(self, vector):
        if self.size == 512:
            vector_matrix = np.tile(np.reshape(vector, (self.size, 1)), self.size)
            encoded_vector = (self.attention_matrix + (vector_matrix/3.14))/2
        elif self.size == 128:
            vector_matrix = np.tile(np.reshape(vector, (self.size, 4)), 32)
            encoded_vector = (self.attention_matrix + (vector_matrix/3.14))/2
        else:
            raise RuntimeError("Currently only image sizes of 512 and 128 are supported")
        
        return encoded_vector.clip(-1, 1)
    
    def __len__(self):
        return len(self.eg3d_data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_path = os.path.join(self.data_dir, self.eg3d_data.iloc[idx, 0])
        image = Image.open(img_path)
        
        latent_vector = self.eg3d_data.iloc[idx, 1]
        
        if self.encode:
            encoded_vector = self.encode_latent_vector(latent_vector)
            item = {'images': image, 'encoded_vectors': torch.tensor(encoded_vector, dtype=torch.float32).unsqueeze(0)}
        else:
            item = {'images': image,'latent_vectors': torch.tensor(latent_vector, dtype=torch.float32)}
        
        if self.transform:
            item['images'] = self.transform(item['images'])
            if type(item) == torch.Tensor:
                item = item.type(torch.float32)
            
        return item

class EG3DImageProcessor(object):
    def __init__(self):
        # self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.processor = CLIPImageProcessor(size=512)
        
    def __call__(self, sample):
        image = sample
        
        image = self.processor(images=image, return_tensors="pt")['pixel_values'][0]
        return image