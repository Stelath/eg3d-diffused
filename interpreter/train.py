import os
import time
import numpy as np
import pandas as pd
from itertools import chain
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from eg3d_dataset import EG3DDataset
from diffuser_utils.evaluate import evaluate_encoder, evaluate, evaluate_ae

import lightning.pytorch as pl
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import FSDPStrategy

from accelerate import Accelerator, DistributedType
from diffusers import UNet1DModel
from diffusers import DPMSolverMultistepScheduler, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

from autoencoder import AutoencoderKLConfig, AutoencoderKL
from transformers import CLIPModel, CLIPVisionModelWithProjection, CLIPConfig, CLIPTextConfig, CLIPVisionConfig

from eg3d_pipeline import EG3DPipeline
from eg3d_loss import EG3DLoss
from eg3d import EG3D

from dataclasses import dataclass

@dataclass
class TrainingConfig:
    rgb = True
    image_size = 512  # the generated image resolution
    # train_batch_size = 64 # 80 for diffuser
    # eval_batch_size = 12  # how many images to sample during evaluation
    train_batch_size = 1
    eval_batch_size = 1
    num_dataloader_workers = 8  # how many subprocesses to use for data loading
    epochs = 500
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    scheduler_timesteps = 1000
    save_image_epochs = 1
    save_model_epochs = 1
    mixed_precision = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    
    train_model = 'autoencoder' # 'diffusion' or 'autoencoder'
    output_dir = f'/scratch1/korte/eg3d-latent-diffuser'
    
    eg3d_model_path = 'eg3d/eg3d_model/ffhqrebalanced512-128.pkl'
    eg3d_latent_vector_size = 512
    
    data_dir = 'data/'
    model_checkpoint = '' # '/scratch1/korte/eg3d-latent-diffuser/autoencoder/ae-69.pth'

    overwrite_output_dir = True
    seed = 0

def train():    
    config = TrainingConfig()
    
    preprocess = transforms.Compose(
        [
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ]
    )

    dataset = EG3DDataset(data_dir=config.data_dir, transform=preprocess, augmented=False, triplanes=True, one_hot=False)

    train_size = int(len(dataset) * 0.95)
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size], generator=torch.Generator().manual_seed(42))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_dataloader_workers, pin_memory=True)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=config.eval_batch_size, shuffle=False, num_workers=config.num_dataloader_workers, pin_memory=True)
    
    print(f"Loaded Dataloaders")
    print(f"Training on {len(train_dataset)} images, evaluating on {len(eval_dataset)} images")
    
    if config.train_model == 'diffusion':
        ### TRAIN DIFFUSER ###
        eg3d = EG3D(config.eg3d_model_path, device='cpu')
        
        vision_config = CLIPVisionConfig(image_size=config.image_size, projection_dim=512)
        vision_transformer = CLIPVisionModelWithProjection(vision_config)
        
        diffuser = EG3DConditional(
            sample_size=config.eg3d_latent_vector_size,
            in_channels=1,
            out_channels=1,
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channes for each UNet block
            time_embed_dim = 512,
            down_block_types=("DownBlock1D", "DownBlock1D", "AttnDownBlock1D", "DownBlock1D", "AttnDownBlock1D", "DownBlock1D"),
            up_block_types=("UpBlock1D", "AttnUpBlock1D", "UpBlock1D", "AttnUpBlock1D", "UpBlock1D", "UpBlock1D"),
        )
        loss_function = nn.MSELoss()

        noise_scheduler = DDPMScheduler(num_train_timesteps=config.scheduler_timesteps, prediction_type='epsilon')
        optimizer = torch.optim.AdamW(chain(diffuser.parameters(), vision_transformer.paramaters()), lr=config.learning_rate)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * config.epochs),
        )

        train_diffuser_loop(config, diffuser, vision_transformer, noise_scheduler, optimizer, lr_scheduler, loss_function, eg3d, train_dataloader, eval_dataset)
    
    if config.train_model == 'autoencoder':
        ### TRAIN AUTOENCODER ###
        train_path = os.path.join(config.output_dir, 'autoencoder/')
        os.makedirs(train_path, exist_ok=True)
        
        autoencoder_config = AutoencoderKLConfig()
        autoencoder = AutoencoderKL(autoencoder_config)

        # Train Model
        checkpoint_callback = ModelCheckpoint(
            save_top_k=10,
            monitor="val/rec_loss",
            mode="min",
            filename="autoencoder-{epoch:02d}-{val/rec_loss:.2f}",
        )
        fsdp = FSDPStrategy()
        trainer = pl.Trainer(callbacks=[checkpoint_callback], default_root_dir=train_path, accelerator="gpu", strategy=fsdp, precision=16, devices=2, check_val_every_n_epoch=5, max_epochs=300)
        trainer.fit(model=autoencoder, train_dataloaders=train_dataloader, val_dataloaders=eval_dataloader)
        

def train_diffuser_loop(config, model, vision_transformer, noise_scheduler, optimizer, lr_scheduler, loss_function, eg3d, train_dataloader, eval_dataset):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps, 
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs")
    )
    if accelerator.is_main_process:
        accelerator.init_trackers("eg3d_li_diffuser")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the 
    # objects in the same order you gave them to the prepare method.
    model, vision_transformer, optimizer, train_dataloader, lr_scheduler, eg3d.G = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler, eg3d.G
    )
    
    eg3d.update_device()
    
    global_step = 0
    # Now you train the model
    for epoch in range(config.epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        
        torch.cuda.empty_cache()
        model.train()
        
        for step, batch in enumerate(train_dataloader):
            images = batch['images']
            latent_vectors = batch['latent_vectors'].unsqueeze(1)
            #encodings = batch['facenet_encoding'].unsqueeze(1)
            # features = batch['features'].unsqueeze(1)
            
            noise = torch.randn(latent_vectors.shape).to(latent_vectors.device)
            bs = latent_vectors.shape[0]
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=latent_vectors.device).long()
            
            latent_vectors = noise_scheduler.add_noise(latent_vectors, noise, timesteps)
            
            with accelerator.accumulate(model):# as _, with accelerator.accumulate(vision_transformer) as _:
                encodings = vision_transformer(images, return_dict=False)[0]
                print("ENCODING SHAPE:", encodings.shape)
                
                # Predict the noise residual
                # latents_input = torch.cat([latent_vectors, features], dim=1)
                noise_pred = model(latent_vectors, timesteps,  return_dict=False)[0]
                loss = loss_function(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
        
        if accelerator.is_main_process:
            model.eval()
            pipeline = EG3DPipeline(unet=accelerator.unwrap_model(model), vision_transformer=vision_transformer.unwrap_model(vision_transformer), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.epochs - 1:
                eval_loss = evaluate(config, epoch, pipeline, eg3d, loss_function, eval_dataset)
                eval_loss = eval_loss.detach().item()
                logs = {"eval_loss": eval_loss}
                accelerator.log(logs, step=global_step)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.epochs - 1:
                pipeline.save_pretrained(f'{config.output_dir}/diffuser/diffuser_{epoch}')


def train_ae_loop(config, model, opt_ae, opt_disc, train_dataloader, eval_dataset, epoch = 0):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps, 
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs")
    )
    if accelerator.is_main_process:
        accelerator.init_trackers("eg3d_li_autoencoder")
    
    disc = model.loss
    
    model, disc, opt_ae, opt_disc, train_dataloader = accelerator.prepare(
        model, disc, opt_ae, opt_disc, train_dataloader
    )
    
    multi = accelerator.num_processes > 1 and accelerator.distributed_type != DistributedType.DEEPSPEED
    precision = accelerator.mixed_precision
    
    # disc.find_unused_parameters = True
    
    global_step = 0
    # Now you train the model
    for epoch in range(config.epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        
        # torch.cuda.empty_cache()
        model.train()
        
        for step, batch in enumerate(train_dataloader):
            triplanes = batch['triplanes'].to(memory_format=torch.contiguous_format)
            
            with accelerator.accumulate(model):
                reconstructions, posterior = model(triplanes)
                last_layer = model.module.decoder.conv_out.weight if multi else model.decoder.conv_out.weight
                
                aeloss, log_dict_ae = disc(triplanes, reconstructions, posterior, 0, global_step,
                                                last_layer=last_layer, split="train")
                accelerator.backward(aeloss)
                opt_ae.step()
                opt_ae.zero_grad()
            
            with accelerator.accumulate(disc):
                # reconstructions, posterior = model(triplanes)
                # last_layer = model.module.decoder.conv_out.weight if multi else model.decoder.conv_out.weight
                discloss, log_dict_disc = disc(triplanes, reconstructions, posterior, 1, global_step,
                                                last_layer=last_layer, split="train")
                print(discloss)
                accelerator.backward(discloss)
                opt_disc.step()
                opt_disc.zero_grad()
            
            progress_bar.update(1)
            
            logs = {**log_dict_ae, **log_dict_disc}

            for key in logs.keys():
                logs[key] = logs[key].item()

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
        
        if accelerator.is_main_process:
            model.eval()
            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.epochs - 1:
                logs = evaluate_ae(config, model, disc, eval_dataset, global_step, multi)

                for key in logs.keys():
                    logs[key] = logs[key].item()

                accelerator.log(logs, step=global_step)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.epochs - 1:
                accelerator.wait_for_everyone()
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                    'opt_ae_state_dict': opt_ae.state_dict(),
                    'opt_disc_state_dict': opt_disc.state_dict(),
                }, f'{config.output_dir}/autoencoder/ae-{epoch + 1}.pth')


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium') # 'medium' or 'high'
    seed_everything(42, workers=True)
    train()
