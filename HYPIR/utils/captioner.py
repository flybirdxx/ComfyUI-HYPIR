from typing import overload, Literal
import re
import base64
from io import BytesIO

from PIL import Image
import torch
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed


class Captioner:

    def __init__(self, device: torch.device) -> "Captioner":
        self.device = device

    @overload
    def __call__(self, image: Image.Image) -> str: ...


class EmptyCaptioner(Captioner):

    def __call__(self, image: Image.Image) -> str:
        return ""


class GPTCaptioner(Captioner):

    DEFAULT_PROMPT = "Provide a detailed description of this image without exceeding 100 words and without line breaks."

    def __init__(self, api_key, base_url, model):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    @staticmethod
    def pil_image_to_base64(image, format="PNG"):
        buffered = BytesIO()
        image.save(buffered, format=format)
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        return img_base64

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1))
    def get_response(self, base64_image, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            stream=False,
        )
        return response.choices[0].message.content

    def __call__(self, image, prompt=None):
        base64_image = self.pil_image_to_base64(image)
        if prompt is None:
            prompt = self.DEFAULT_PROMPT
        caption = self.get_response(base64_image, prompt=prompt)
        return caption
