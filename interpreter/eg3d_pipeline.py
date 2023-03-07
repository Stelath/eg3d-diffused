import torch

from typing import List, Optional, Tuple, Union

from diffusers.pipeline_utils import DiffusionPipeline
# from diffusers.configuration_utils import FrozenDict
# from diffusers.utils import deprecate

from PIL import Image
from typing import Optional, Tuple, Union
from diffusers.pipelines import DDPMPipeline

class EG3DPipeline(DiffusionPipeline):
    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        encodings,
        features,
        num_inference_steps: int = 1000,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        **kwargs,
    ) -> torch.Tensor:
        #encodings = encodings.to(self.device).unsqueeze(1)
        features = features.to(self.device).unsqueeze(1)
        
        self.scheduler.set_timesteps(num_inference_steps)
        
        # bs = encodings.shape[0]
        latent_vectors = torch.randn(features.shape, device=self.device, generator=generator)
        
        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            latents_input = torch.cat([latent_vectors, features], dim=1)
            latents_input = self.scheduler.scale_model_input(latents_input, t)
            
            model_output = self.unet(latents_input, t).sample

            # 2. compute previous image: x_t -> x_t-1
            latent_vectors = self.scheduler.step(model_output, t, latent_vectors).prev_sample

        latent_vectors = latent_vectors.squeeze(1)
        
        return latent_vectors
