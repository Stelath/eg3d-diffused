import torch
import torch.nn.functional as F
import torchvision.transforms as T

from eg3d import EG3D

class EG3DLoss():
    def __init__(self, model=None, model_path=None, image_size=512, device='cuda'):
        if model == None and model_path == None:
            raise Exception("Must provide model or model_path") 
        
        if model != None:
            self.eg3d = model
        else:
            self.eg3d = EG3D(model_path, device)
        
        self.image_size = image_size
        self.resize = T.Resize((image_size, image_size))
            
    def __call__(self, inputs, targets):
        # Targets are an image/contours of the face
        
        inputs = self.eg3d.generate_imgs(inputs)
        
        inputs = self.resize(inputs)
        # inputs = torch.empty((imgs.shape[0], 3, self.image_size, self.image_size))
        # for i, img in enumerate(imgs):
        #     inputs[i] = self.resize(img)
        
        loss = F.mse_loss(inputs, targets)
        return loss
    
    def get_eg3d(self):
        return self.eg3d
