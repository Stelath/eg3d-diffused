import torch

def get_device(model) -> torch.device:
    return next(model.parameters()).device
    
def get_batch(dataset, start, end) -> torch.Tensor:
    batch = {}
    first_item = dataset[start]
    for k in first_item.keys():
        batch[k] = torch.empty((end - start,) + tuple(first_item[k].shape))
    
    for i in range(start + 1, end):
        temp = dataset[i]
        for k in temp.keys():
            batch[k][i - start] = temp[k]
    
    return batch
