import torch
from torch.utils.data import Dataset, DataLoader
from eg3d_dataset import EG3DDataset
from diffusers import UNet2DModel

from dataclasses import dataclass

@dataclass
class TrainingConfig:
    image_size = 128  # the generated image resolution
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    num_dataloader_workers = 4  # how many subprocesses to use for data loading
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = 'ddpm-eg3d-latent-interpreter'
    
    data_dir = 'data/'
    csv_file = 'eg3d_data.csv'

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False  
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0



def train():    
    config = TrainingConfig()
    
    dataset = EG3DDataset(csv_file=config.csv_file, data_dir=config.data_dir)
    
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_dataloader_workers)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_dataloader_workers)
    
    model = UNet2DModel(
        sample_size=config.image_size,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channes for each UNet block
        down_block_types=( 
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ), 
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D"
        ),
    )


