import numpy as np
from collections import OrderedDict
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from distributions import DiagonalGaussianDistribution

from transformers import CLIPImageProcessor, CLIPVisionModel
from autoencoder import AutoencoderKL
from diffusers import UNet2DConditionModel, PNDMScheduler

from deepspeed.ops.adam import DeepSpeedCPUAdam

import lightning.pytorch as pl
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from torch.distributed.fsdp.wrap import wrap

class TRIAD(pl.LightningModule):
    def __init__(self, aec_pth, scheduler_timesteps=1000):
        super().__init__()
        self.save_hyperparameters()
        self.fsdp = False
        
        self.vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        self.vision_encoder_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        
        self.ae = AutoencoderKL.load_from_checkpoint(aec_pth, strict=False)
        self.ae.freeze()
        
        self.noise_scheduler = PNDMScheduler(num_train_timesteps=scheduler_timesteps, prediction_type='epsilon')
        self.diffuser = UNet2DConditionModel(
            sample_size=64,
            in_channels=128,
            out_channels=128,
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            #block_out_channels=(1280, 2560, 5120, 5120),  # the number of output channes for each UNet block
            block_out_channels=(320, 640, 1280, 1280),
            down_block_types = ('CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'DownBlock2D'),
            up_block_types = ('UpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D'),
            cross_attention_dim=1024,
        )
    
    def forward(self, sample, timesteps, encoder_hidden_states, **kwargs):
        pred_noise = self.diffuser(sample, timesteps, encoder_hidden_states, **kwargs)
        
        return pred_noise
        
    
    @torch.no_grad()
    def encode_images(self, images):
        # inputs = self.vision_encoder_processor(images=images, return_tensors="pt")
        inputs = images
        encodings = self.vision_encoder(pixel_values=inputs, return_dict=True)
        encodings = encodings.last_hidden_state
        
        return encodings
    
    @torch.no_grad()
    def encode_triplanes(self, triplanes):
        encodings = self.ae.encode(triplanes)
        
        return encodings
    
    def training_step(self, batch, batch_idx):
        images = batch['images']
        triplanes = batch['triplanes']
        
        encoded_images = self.encode_images(images)
        
        encoded_triplanes = self.encode_triplanes(triplanes)
        encoded_triplanes = encoded_triplanes.sample()

        noise = torch.randn(encoded_triplanes.shape).to(encoded_triplanes.device)
        bs = encoded_triplanes.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bs,), device=encoded_triplanes.device).long()

        encoded_triplanes = self.noise_scheduler.add_noise(encoded_triplanes, noise, timesteps)
        
        pred_noise = self(encoded_triplanes, timesteps, encoded_images, return_dict=True)[0]
        
        loss = F.mse_loss(pred_noise, noise)
        
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        
        return loss
    
    def configure_sharded_model(self):
        self.diffuser = wrap(self.diffuser)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.diffuser.parameters(), lr=1e-4)
        # optimizer = DeepSpeedCPUAdam(self.diffuser.parameters(), lr=1e-4)
        
        return optimizer
    
    @torch.no_grad()
    def gen_triplanes(self, images, num_triplanes = 4, triplane_shape = (256, 256), num_timesteps=1000):
        self.noise_scheduler.set_timesteps(num_timesteps)
        encoded_images = self.encode_images(images.to(self.device))
        
        triplane_shape = tuple(int(s/4) for s in triplane_shape)
        latents = torch.randn((num_triplanes, 128, *triplane_shape)).to(self.device)
        
        for t in tqdm(self.noise_scheduler.timesteps):
            pred_noise = self(latents, t, encoded_images).sample
            latents = self.noise_scheduler.step(pred_noise, t, latents).prev_sample
        
        triplanes = self.ae.decoder(latents)
        
        return triplanes
    
