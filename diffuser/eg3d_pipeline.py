import torch

from diffusers.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.configuration_utils import FrozenDict
from diffusers.utils import deprecate

from PIL import Image
from typing import Optional, Tuple, Union

# Refactor type name
ImagePipelineInput = ImagePipelineOutput

class EG3DPipeline(DiffusionPipeline):
    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        images: ImagePipelineInput,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        predict_epsilon = deprecate("predict_epsilon", "0.11.0", message, take_from=kwargs)

        if predict_epsilon is not None:
            new_config = dict(self.scheduler.config)
            new_config["prediction_type"] = "epsilon" if predict_epsilon else "sample"
            self.scheduler._internal_dict = FrozenDict(new_config)

        if generator is not None and generator.device.type != self.device.type and self.device.type != "mps":
            message = (
                f"The `generator` device is `{generator.device}` and does not match the pipeline "
                f"device `{self.device}`, so the `generator` will be ignored. "
                f'Please use `torch.Generator(device="{self.device}")` instead.'
            )
            deprecate(
                "generator.device == 'cpu'",
                "0.11.0",
                message,
            )
            generator = None
        
        if self.unet.sample_size == images.shape[1]:
            raise RuntimeError("The input images are not the correct size for the given UNet")

        images = images.to(self.device)
        self.scheduler.set_timesteps(num_inference_steps)
        
        for t in self.progress_bar(num_inference_steps):
            # 1. predict noise model_output
            model_output = self.unet(images, t).sample

            # 2. compute previous image: x_t -> x_t-1
            images = self.scheduler.step(model_output, t, images, generator=generator).prev_sample

        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            images = self.numpy_to_pil(images)

        if not return_dict:
            return (images,)

        return ImagePipelineOutput(images=images)