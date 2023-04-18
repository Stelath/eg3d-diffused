import os
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from diffuser_utils.encoding import create_attention_matrix

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import BlipProcessor, CLIPImageProcessor

class EG3DDataset(Dataset):
    def __init__(self, data_dir, image_size=512, transform=None, triplanes=False, latent_triplanes=False, face_landmarks=False, one_hot=True):
        self.eg3d_data = pd.read_pickle(os.path.join(data_dir, 'dataset.df')) if not face_landmarks else pd.read_pickle(os.path.join(data_dir, 'augmented_dataset.df'))
        self.data_dir = data_dir
        self.transform = transform
        self.size = image_size
        self.attention_matrix = create_attention_matrix(self.size, self.size)
        self.has_triplanes = triplanes
        self.has_latent_triplanes = latent_triplanes
        self.face_landmarks = face_landmarks
        self.one_hot = one_hot
        
        if triplanes:
            self.triplanes_memmap = np.memmap(os.path.join(data_dir, 'triplanes.mmap'), dtype='float32', mode='r', shape=(len(self.eg3d_data.index), 96, 256, 256))
        
        self.feature_keys = [{'Woman': 0, 'Man': 1}, {'asian': 0, 'indian': 1, 'black': 2, 'white': 3, 'middle eastern': 4, 'latino hispanic': 5}, {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprise': 5, 'neutral': 6}]
    
    def __len__(self):
        return len(self.eg3d_data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_path = os.path.join(self.data_dir, self.eg3d_data.iloc[idx, 0])
        image = Image.open(img_path)
        
        latent_vector = self.eg3d_data.iloc[idx, 1]
        
        item = None
        if self.has_triplanes:
            item = self.get_triplanes(idx, image, latent_vector)
        
        if self.face_landmarks:
            item = self.get_face_landmarks(idx, image, latent_vector)
        
        if item == None:
            item = {'images': image,'latent_vectors': torch.tensor(latent_vector, dtype=torch.float32)}
        
        if self.transform:
            item['images'] = self.transform(item['images'])
            if type(item) == torch.Tensor:
                item = item.type(torch.float32)
            
        return item
    
    def get_triplanes(self, idx, image, latent_vector):
        triplanes = self.triplanes_memmap[self.eg3d_data.iloc[idx, 2]]
            
        if self.has_latent_triplanes:
            with open(os.path.join(self.data_dir, 'encoded_triplanes', f'{self.eg3d_data.iloc[idx, 2]:04d}.pkl'), 'rb') as lt:
                latent_triplane = pickle.load(lt)
            item = {'images': image,'latent_vectors': torch.tensor(latent_vector, dtype=torch.float32), 'triplanes': torch.tensor(triplanes), 'latent_triplanes': latent_triplane}
        else:
            item = {'images': image,'latent_vectors': torch.tensor(latent_vector, dtype=torch.float32), 'triplanes': torch.tensor(triplanes)}
        
        return item
    
    def get_face_landmarks(self, idx, image, latent_vector):
        analysis = self.eg3d_data.iloc[idx]['analysis']
        landmarks = self.eg3d_data.iloc[idx]['landmarks']
        encoding = self.eg3d_data.iloc[idx]['encoding']

        if self.one_hot:
            age = torch.tensor([analysis['age']])
            gender = F.one_hot(torch.tensor(self.feature_keys[0][analysis['dominant_gender']], dtype=torch.long), num_classes=2)
            race = F.one_hot(torch.tensor(self.feature_keys[1][analysis['dominant_race']], dtype=torch.long), num_classes=6)
            emotion = F.one_hot(torch.tensor(self.feature_keys[2][analysis['dominant_emotion']], dtype=torch.long), num_classes=7)

            features = torch.zeros((512), dtype=torch.float32)
            features[0] = age
            features[1:3] = gender
            features[3:9] = race
            features[9:16] = emotion
            features[16:212] = torch.from_numpy(landmarks.flatten())
        else:
            gender = torch.tensor(self.feature_keys[0][analysis['dominant_gender']], dtype=torch.long)
            race = torch.tensor(self.feature_keys[1][analysis['dominant_race']] + 2, dtype=torch.long)
            emotion = torch.tensor(self.feature_keys[2][analysis['dominant_emotion']] + (2+6), dtype=torch.long)
            age = torch.tensor(analysis['age'] + (2+6+7), dtype=torch.long)

            features = torch.cat([gender,race,emotion,age])

        item = {'images': image,'latent_vectors': torch.tensor(latent_vector, dtype=torch.float32), 'facenet_encoding': torch.tensor(encoding, dtype=torch.float32), 'features': features}
        
        return item

class EG3DImageProcessor(object):
    def __init__(self):
        # self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
    def __call__(self, sample):
        image = sample
        
        image = self.processor(images=image, return_tensors="pt")['pixel_values'][0]
        return image