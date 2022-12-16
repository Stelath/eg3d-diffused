import os

import torch
from PIL import Image

def make_grid(images, rows, cols):
    w, h = images[0].size
    images[0].save("test.png")
    print("WIDTH AND HEIGHT")
    print(w)
    print(h)
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def evaluate(config, epoch, pipeline, eval_dataloader, save=True):
    batch = next(iter(eval_dataloader))
    
    images = pipeline(
        batch['images'],
    ).images

    comparisons = []
    for i in range(config.eval_batch_size):
        comparison = make_grid([images[i], batch['encoded_vectors']], rows=1, cols=2)
        comparisons.append(comparison)
    
    if save:
        for i, image in enumerate(comparisons):
            # Save the images
            eval_dir = os.path.join(config.output_dir, "samples")
            os.makedirs(eval_dir, exist_ok=True)
            image.save(f"{eval_dir}/{epoch:04d}_{i:02d}.png")
    else:
        return comparisons
