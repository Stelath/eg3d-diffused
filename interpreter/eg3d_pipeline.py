import torch

from diffusers.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.configuration_utils import FrozenDict
from diffusers.utils import deprecate

from PIL import Image
from typing import Optional, Tuple, Union
from diffusers.pipelines import DDPMPipeline

# Refactor type name
ImagePipelineInput = ImagePipelineOutput

class EG3DPipeline(DiffusionPipeline):
    def __init__(self, encoder, unet, scheduler):
        super().__init__()
        # self.encoder = encoder
        self.register_modules(encoder=encoder, unet=unet, scheduler=scheduler)
        # self.encoder.eval()

    @torch.no_grad()
    def __call__(
        self,
        images: ImagePipelineInput,
        num_inference_steps: int = 60,
        **kwargs,
    ) -> torch.Tensor:
        images = images.to(self.device)
        
        self.scheduler.set_timesteps(num_inference_steps)
        
        bs = images.shape[0]
        
        latent_vectors = self.encoder(images)
        latent_vectors = latent_vectors.detach()
        latent_vectors = latent_vectors.unsqueeze(1)
        del images
        
        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.unet(latent_vectors, t).sample

            # 2. compute previous image: x_t -> x_t-1
            latent_vectors = self.scheduler.step(model_output, t, latent_vectors).prev_sample

        latent_vectors = latent_vectors.squeeze(1)    
        
        return latent_vectors
