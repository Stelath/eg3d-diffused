import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from distributions import DiagonalGaussianDistribution
from lpips_discriminator import LPIPSWithDiscriminator

from transformers import AutoProcessor, CLIPVisionModel
from autoencoder import AutoencoderKL
from diffusers import UNet2DConditionModel

import lightning.pytorch as pl
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from torch.distributed.fsdp.wrap import wrap

class TRIAD(pl.LightningModule):
    def __init__(self, aec_pth):
        super().__init__()
        self.save_hyperparameters()
        
        self.vt = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.vt_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.ae = AutoencoderKL.load_from_checkpoint(ckpt_pth, strict=False)
        self.diffuser = UNet2DConditionModel(
            sample_size=64,
            in_channels=128,
            out_channels=128,
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(320, 640, 1280, 1280),  # the number of output channes for each UNet block
            down_block_types = ('CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'DownBlock2D'),
            up_block_types = ('UpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D'),
        )
    
    def forward(self, batch, batch_idx):
        images = batch['images']
        triplanes = batch['triplanes']
        
        encoded_images = self.encode_images(images)
        
        triplanes_gaussian_dist = self.encode_triplanes(tripanes)
        encoded_triplanes = triplanes_gaussian_dist.sample()
        
        noise = torch.randn(latent_vectors.shape).to(latent_vectors.device)
        bs = encoded_triplanes.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bs,), device=latent_vectors.device).long()

        latent_vectors = self.noise_scheduler.add_noise(encoded_triplanes, noise, timesteps)
        
        
        
    
    @torch.with_no_grad()
    def encode_images(self, images):
        inputs = processor(images=images, return_tensors="pt")
        encodings = model(**inputs)
        
        return encodings
    
    @torch.with_no_grad()
    def encode_triplanes(self, triplanes):
        encodings = self.ae.encoder(triplanes)
        
        return encodings
    
    def training_step(self, batch, batch_idx):
        encodings
