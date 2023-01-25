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
        self.register_modules(encoder=encoder, unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        images: ImagePipelineInput,
        num_inference_steps: int = 1000,
        **kwargs,
    ) -> torch.Tensor:
        message = (
            "Please make sure to instantiate your scheduler with `prediction_type` instead. E.g. `scheduler ="
            " DDPMScheduler.from_pretrained(<model_id>, prediction_type='epsilon')`."
        )
        predict_epsilon = deprecate("predict_epsilon", "0.11.0", message, take_from=kwargs)

        if predict_epsilon is not None:
            new_config = dict(self.scheduler.config)
            new_config["prediction_type"] = "epsilon" if predict_epsilon else "sample"
            self.scheduler._internal_dict = FrozenDict(new_config)

        images = images.to(self.device)
        
        self.scheduler.set_timesteps(num_inference_steps)
        
        latent_vectors = self.encoder(images)
        del images
        
        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.unet(latent_vectors, t).sample

            # 2. compute previous image: x_t -> x_t-1
            latent_vectors = self.scheduler.step(model_output, t, latent_vectors).prev_sample

        return latent_vectors
