import torch
import torch.nn.functional as F

from eg3d import EG3D

class EG3DLoss():
    def __init__(self, model_path, device='cuda'):
        self.eg3d = EG3D(model_path, device)
        
    def __call__(self, inputs, targets):
        # Targets are an image/contours of the face
        
        inputs = self.eg3d.generate_imgs(inputs)
        
        loss = F.mse_loss(inputs, targets)
        return loss
