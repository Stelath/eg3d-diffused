import os

import torch
import numpy as np
import pandas as pd
from PIL import Image

from .utils import get_device

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def evaluate_encoder(config, epoch, model, eg3d, eval_dataset):
    random_slice = np.random.randint(len(eval_dataset) - config.eval_batch_size)
    batch = eval_dataset[random_slice:random_slice + config.eval_batch_size]
    imgs = batch['images'].to(get_device(model))
    
    with torch.no_grad():
        predicted_lvs = model(imgs).cpu().numpy()
    
    imgs = imgs.cpu().numpy().transpose(0, 2, 3, 1)
    predicted_imgs = eg3d.generate_imgs(predicted_lvs, transpose=True).cpu().numpy()
    
    comparisons = []
    for i in range(config.eval_batch_size):
        comparison = make_grid([imgs[i], predicted_imgs[i]], rows=1, cols=2)
        comparisons.append(comparison)
    
    for i, image in enumerate(comparisons):
        # Save the images
        eval_dir = os.path.join(config.output_dir, "samples")
        os.makedirs(eval_dir, exist_ok=True)
        image.save(f"{eval_dir}/{epoch:04d}_{i:02d}.png")

def evaluate(config, epoch, pipeline, eg3d, eval_dataset):
    random_slice = np.random.randint(len(eval_dataset) - config.eval_batch_size)
    batch = eval_dataset[random_slice:random_slice + config.eval_batch_size]
    imgs = batch['images']
    
    predicted_lvs = pipeline(imgs)

    imgs = imgs.cpu().numpy().transpose(0, 2, 3, 1)
    predicted_imgs = eg3d.generate_imgs(predicted_lvs, transpose=True).cpu().numpy()

    comparisons = []
    for i in range(config.eval_batch_size):
        comparison = make_grid([imgs[i], predicted_imgs[i]], rows=1, cols=2)
        comparisons.append(comparison)
    
    for i, image in enumerate(comparisons):
        # Save the images
        eval_dir = os.path.join(config.output_dir, "samples")
        os.makedirs(eval_dir, exist_ok=True)
        image.save(f"{eval_dir}/{epoch:04d}_{i:02d}.png")

def vision_evaluate(config, epoch, model, eval_dataloader, device='cuda', render=False, save=True):
    model.eval()
    batch = next(iter(eval_dataloader))
    images = [image for image in batch['images']]
    latent_vectors = batch['latent_vectors'].cpu().numpy()
    
    with torch.no_grad():
        latent_vectors_pred = model(images).cpu().numpy()
    
    if save:
        df = pd.DataFrame(columns = ['latent_vectors', 'latent_vectors_pred'])
        for i in range(len(latent_vectors)):
            df.loc[len(df.index)] = [latent_vectors[i], latent_vectors_pred[i]]
        eval_dir = os.path.join(config.output_dir, "samples")
        os.makedirs(eval_dir, exist_ok=True)
        df.to_pickle(f"{eval_dir}/{epoch:04d}.df")
