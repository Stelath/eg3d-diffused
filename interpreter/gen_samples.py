import os

import torch
import numpy as np
import pandas as pd
from PIL import Image

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def diffusion_evaluate(config, epoch, pipeline, eval_dataloader, save=True):
    batch = next(iter(eval_dataloader))
    images = batch['images']
    encoded_vectors = [Image.fromarray(np.squeeze(img)) for img in ((batch['encoded_vectors'].permute(0, 2, 3, 1) + 1.0) * 127.5).type(torch.uint8).numpy()]
    
    images = pipeline(
        images,
    ).images

    comparisons = []
    for i in range(config.eval_batch_size):
        comparison = make_grid([images[i], encoded_vectors[i]], rows=1, cols=2)
        comparisons.append(comparison)
    
    if save:
        for i, image in enumerate(comparisons):
            # Save the images
            eval_dir = os.path.join(config.output_dir, "samples")
            os.makedirs(eval_dir, exist_ok=True)
            image.save(f"{eval_dir}/{epoch:04d}_{i:02d}.png")
    else:
        return comparisons

def vision_evaluate(config, epoch, model, eval_dataloader, device='cuda', render=False, save=True):
    model.eval()
    batch = next(iter(eval_dataloader))
    images = [image for image in batch['images']]
    latent_vectors = batch['latent_vectors'].cpu().numpy()
    
    with torch.no_grad():
        latent_vectors_pred = model(images).cpu().numpy()

    # comparisons = []
    # for i in range(config.eval_batch_size):
    #     comparison = make_grid([images[i], encoded_vectors[i]], rows=1, cols=2)
    #     comparisons.append(comparison)
    
    # if save:
    #     for i, image in enumerate(comparisons):
    #         eval_dir = os.path.join(config.output_dir, "samples")
    #         os.makedirs(eval_dir, exist_ok=True)
    #         image.save(f"{eval_dir}/{epoch:04d}_{i:02d}.png")
    if save:
        df = pd.DataFrame(columns = ['latent_vectors', 'latent_vectors_pred'])
        for i in range(len(latent_vectors)):
            df.loc[len(df.index)] = [latent_vectors[i], latent_vectors_pred[i]]
        eval_dir = os.path.join(config.output_dir, "samples")
        os.makedirs(eval_dir, exist_ok=True)
        df.to_pickle(f"{eval_dir}/{epoch:04d}.df")
