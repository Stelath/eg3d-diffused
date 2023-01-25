import torch

def get_device(model) -> torch.device:
        return next(model.parameters()).device