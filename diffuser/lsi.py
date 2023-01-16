import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from accelerate import Accelerator

import torch
from torch import nn
from torchvision import transforms
from eg3d_dataset import EG3DDataset
from gen_samples import vision_evaluate
from torchvision.models import convnext_base, convnext_small

from eg3d_loss import EG3DLoss
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    image_size = 512  # the generated image resolution
    train_batch_size = 8
    eval_batch_size = 8  # how many images to sample during evaluation
    num_dataloader_workers = 12  # how many subprocesses to use for data loading
    num_epochs = 60
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    scheduler_train_timesteps = 1000
    eval_inference_steps = 1000
    save_image_epochs = 10
    save_model_epochs = 10
    mixed_precision = 'no'  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = 'eg3d-latent-interpreter'
    
    eg3d_model_path = 'eg3d/eg3d_model/ffhqrebalanced512-128.pkl'
    data_dir = 'data_color/'
    df_file = 'dataset.df'

    overwrite_output_dir = True
    seed = 0

config = TrainingConfig()

preprocess = transforms.Compose(
        [
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

dataset = EG3DDataset(df_file=config.df_file, data_dir=config.data_dir, transform=preprocess, encode=False)

train_size = int(len(dataset) * 0.95)
eval_size = len(dataset) - train_size
train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_dataloader_workers)
eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=config.eval_batch_size, shuffle=True, num_workers=config.num_dataloader_workers)

model = convnext_small(num_classes=512)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
vector_loss_function = nn.SmoothL1Loss(reduction='mean')
eg3d_loss_function = EG3DLoss(config.eg3d_model_path)


def train_loop(config, model, optimizer, vector_loss_function, eg3d_loss_function, train_dataloader, eval_dataloader):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps, 
        log_with="tensorboard",
        logging_dir=os.path.join(config.output_dir, "logs")
    )
    if accelerator.is_main_process:
        accelerator.init_trackers("eg3d_latent_interpreter")

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    global_step = 0
    
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        model.train()
        
        for step, batch in enumerate(train_dataloader):
            images = batch['images']
            latent_vectors = batch['latent_vectors']
            
            with accelerator.accumulate(model):
                latent_vectors_pred = model(images)
                
                loss = vector_loss_function(latent_vectors_pred, latent_vectors) + eg3d_loss_function(latent_vectors_pred, images)
                accelerator.backward(loss)

                optimizer.step()
                # lr_scheduler.step()
                optimizer.zero_grad()
                
            progress_bar.update(1)
            logs = {"train_loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
        
        model.eval()
        avg_eval_loss = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                latent_vectors_pred = model(pixel_values=images).pooler_output
                
                loss = loss_function(latent_vectors_pred, latent_vectors)
                avg_eval_loss.append(loss.detach().item())
        avg_eval_loss = sum(avg_eval_loss) / len(avg_eval_loss)
        logs = {"eval_loss": avg_eval_loss}
        accelerator.log(logs, step=global_step)

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            # if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
            #     render = epoch == config.num_epochs - 1
            #     vision_evaluate(config, epoch, model, eval_dataloader, render=render)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, os.path.join(config.output_dir, 'model.pth'))

if __name__ == '__main__':
    train_loop(config, model, optimizer, vector_loss_function, eg3d_loss_function, train_dataloader, eval_dataloader)