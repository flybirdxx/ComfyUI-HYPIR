# Prediction interface for Cog ⚙️
# https://cog.run/python

import random

from cog import BasePredictor, Input, Path
from PIL import Image
from torchvision.transforms.functional import to_tensor
from accelerate.utils import set_seed

from HYPIR.enhancer.sd2 import SD2Enhancer


class Predictor(BasePredictor):

    def setup(self) -> None:
        self.model = SD2Enhancer(
            base_model_path="cog_im_files/sd2_diffusers",
            weight_path="cog_im_files/HYPIR_sd2.pth",
            lora_modules=[
                "to_k",
                "to_q",
                "to_v",
                "to_out.0",
                "conv",
                "conv1",
                "conv2",
                "conv_shortcut",
                "conv_out",
                "proj_in",
                "proj_out",
                "ff.net.2",
                "ff.net.0.proj",
            ],
            lora_rank=256,
            model_t=200,
            coeff_t=200,
            device="cuda",
        )
        self.model.init_models()

    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(description="Prompt", default=""),
        upscale: float = Input(description="Upscale Factor", ge=1, le=8, default=1),
        seed: int = Input(description="Random seed", default=-1),
    ) -> Path:
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        set_seed(seed)

        output = "/tmp/out.png"
        image = Image.open(str(image)).convert("RGB")
        image_tensor = to_tensor(image).unsqueeze(0)
        result = self.model.enhance(
            lq=image_tensor,
            prompt=prompt,
            upscale=upscale,
            return_type="pil",
        )[0]
        result.save(output)
        return Path(output)
