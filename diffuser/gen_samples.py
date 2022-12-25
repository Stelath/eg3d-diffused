import os

import torch
import numpy as np
from PIL import Image

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def evaluate(config, epoch, pipeline, eval_dataloader, save=True):
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
