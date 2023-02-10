import os
import time
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from eg3d_dataset import EG3DDataset
from diffuser_utils.evaluate import evaluate_encoder, evaluate
from torchvision.models import convnext_base, convnext_small, efficientnet_v2_s

from accelerate import Accelerator
from diffusers import UNet1DModel
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

from eg3d_pipeline import EG3DPipeline
from eg3d_encoder import EG3DEncoder
from eg3d_loss import EG3DLoss
from eg3d import EG3D

from dataclasses import dataclass

@dataclass
class TrainingConfig:
    rgb = True
    image_size = 512  # the generated image resolution
    train_batch_size = 40 # 80 for diffuser
    eval_batch_size = 12  # how many images to sample during evaluation
    num_dataloader_workers = 12  # how many subprocesses to use for data loading
    encoder_epochs = 120
    diffuser_epochs = 400
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    scheduler_train_timesteps = 60
    eval_inference_steps = 60
    save_image_epochs = 1
    save_model_epochs = 2
    mixed_precision = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = 'eg3d-latent-diffusion'
    
    eg3d_model_path = 'eg3d/eg3d_model/ffhqrebalanced512-128.pkl'
    eg3d_latent_vector_size = 512
    
    data_dir = 'data_color/'
    df_file = 'dataset.df'

    overwrite_output_dir = True
    seed = 0

def train():    
    config = TrainingConfig()
    
    preprocess = transforms.Compose(
        [
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
        ]
    )

    dataset = EG3DDataset(df_file=config.df_file, data_dir=config.data_dir, transform=preprocess, encode=False)

    train_size = int(len(dataset) * 0.95)
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size], generator=torch.Generator().manual_seed(42))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_dataloader_workers)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=config.eval_batch_size, shuffle=True, num_workers=config.num_dataloader_workers)
    
    print(f"Loaded Dataloaders")
    print(f"Training on {len(train_dataset)} images, evaluating on {len(eval_dataset)} images")
    
    eg3d = EG3D(config.eg3d_model_path, device='cpu')
    vector_loss_function = nn.L1Loss(reduction='mean')
    
    ### TRAIN DIFFUSER ###
    encoder = efficientnet_v2_s(num_classes=512)
    encoder.load_state_dict(torch.load(config.pretrained_encoder)['model_state_dict'])

    diffuser = UNet1DModel(
        sample_size=config.eg3d_latent_vector_size,  # the target image resolution
        in_channels=1,
        out_channels=1,
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(128, 128, 256, 256, 512),  # the number of output channes for each UNet block
        down_block_types=("DownBlock1D", "DownBlock1D", "DownBlock1D", "AttnDownBlock1D", "DownBlock1D"),
        up_block_types=("UpBlock1D", "AttnUpBlock1D", "UpBlock1D", "UpBlock1D", "UpBlock1D"),
    )

    noise_scheduler = DDPMScheduler(num_train_timesteps=config.scheduler_train_timesteps, prediction_type='epsilon')
    optimizer = torch.optim.AdamW(diffuser.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.diffuser_epochs),
    )

    encoder.eval()
    train_diffuser_loop(config, diffuser, encoder, noise_scheduler, optimizer, lr_scheduler, vector_loss_function, eg3d, train_dataloader, eval_dataset)

def train_diffuser_loop(config, model, encoder, noise_scheduler, optimizer, lr_scheduler, loss_function, eg3d, train_dataloader, eval_dataset):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps, 
        log_with="tensorboard",
        logging_dir=os.path.join(config.output_dir, "logs")
    )
    if accelerator.is_main_process:
        accelerator.init_trackers("eg3d_li_diffuser")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the 
    # objects in the same order you gave them to the prepare method.
    model, encoder, optimizer, train_dataloader, lr_scheduler, eg3d.G = accelerator.prepare(
        model, encoder, optimizer, train_dataloader, lr_scheduler, eg3d.G
    )
    
    eg3d.update_device()
    
    encoder.eval()
    
    global_step = 0
    # Now you train the model
    for epoch in range(config.diffuser_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        
        torch.cuda.empty_cache()
        
        for step, batch in enumerate(train_dataloader):
            images = batch['images']
            
            bs = images.shape[0]
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=images.device).long()
            
            latent_vectors = batch['latent_vectors']
            predicted_latent_vectors = encoder(images)
            
            noise = latent_vectors - predicted_latent_vectors
            del predicted_latent_vectors
            
            noise = noise.unsqueeze(1)
            latent_vectors = latent_vectors.unsqueeze(1)
            # predicted_latent_vectors = predicted_latent_vectors.unsqueeze(1)
            
            latent_vectors = noise_scheduler.add_noise(latent_vectors, noise, timesteps)
            
            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(latent_vectors, timesteps, return_dict=False)[0]
                loss = loss_function(noise_pred, noise)
                accelerator.backward(loss)

                print("STARTING FINAL BIT")
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                print("CLIPPED NORM")
                optimizer.step()
                lr_scheduler.step()
                print("STEPPED MODEL")
                optimizer.zero_grad()
                print("ZERO GRAD")
            
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            model.eval()
            pipeline = EG3DPipeline(encoder=encoder, unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.diffuser_epochs - 1:
                eval_loss = evaluate(config, epoch, pipeline, eg3d, loss_function, eval_dataset)
                eval_loss = eval_loss.detach().item()
                logs = {"eval_loss": eval_loss}
                accelerator.log(logs, step=global_step)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.diffuser_epochs - 1:
                pipeline.save_pretrained(f'{config.output_dir}/diffuser/diffuser_{epoch}')
            

if __name__ == "__main__":
    train()
