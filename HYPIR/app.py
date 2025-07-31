import random
import os
from argparse import ArgumentParser

import gradio as gr
import torchvision.transforms as transforms
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from dotenv import load_dotenv
from PIL import Image

from HYPIR.enhancer.sd2 import SD2Enhancer
from HYPIR.utils.captioner import GPTCaptioner


load_dotenv()
error_image = Image.open(os.path.join("assets", "gradio_error_img.png"))

parser = ArgumentParser()
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--local", action="store_true")
parser.add_argument("--port", type=int, default=7860)
parser.add_argument("--gpt_caption", action="store_true")
parser.add_argument("--max_size", type=str, default=None, help="Comma-seperated image size")
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()

max_size = args.max_size
if max_size is not None:
    max_size = tuple(int(x) for x in max_size.split(","))
    if len(max_size) != 2:
        raise ValueError(f"Invalid max size: {max_size}")
    print(f"Max size set to {max_size}, max pixels: {max_size[0] * max_size[1]}")

if args.gpt_caption:
    if (
        "GPT_API_KEY" not in os.environ
        or "GPT_BASE_URL" not in os.environ
        or "GPT_MODEL" not in os.environ
    ):
        raise ValueError(
            "If you want to use gpt-generated caption, "
            "please specify both `GPT_API_KEY`, `GPT_BASE_URL` and `GPT_MODEL` in your .env file. "
            "See README.md for more details."
        )
    captioner = GPTCaptioner(
        api_key=os.getenv("GPT_API_KEY"),
        base_url=os.getenv("GPT_BASE_URL"),
        model=os.getenv("GPT_MODEL"),
    )
to_tensor = transforms.ToTensor()

config = OmegaConf.load(args.config)
if config.base_model_type == "sd2":
    model = SD2Enhancer(
        base_model_path=config.base_model_path,
        weight_path=config.weight_path,
        lora_modules=config.lora_modules,
        lora_rank=config.lora_rank,
        model_t=config.model_t,
        coeff_t=config.coeff_t,
        device=args.device,
    )
    model.init_models()
else:
    raise ValueError(config.base_model_type)


def process(
    image,
    prompt,
    upscale,
    seed,
    progress=gr.Progress(track_tqdm=True),
):
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
    set_seed(seed)
    image = image.convert("RGB")
    # Check image size
    if max_size is not None:
        out_w, out_h = tuple(int(x * upscale) for x in image.size)
        if out_w * out_h > max_size[0] * max_size[1]:
            return error_image, (
                "Failed: The requested resolution exceeds the maximum pixel limit. "
                f"Your requested resolution is ({out_h}, {out_w}). "
                f"The maximum allowed pixel count is {max_size[0]} x {max_size[1]} "
                f"= {max_size[0] * max_size[1]} :("
            )
    if prompt == "auto":
        if args.gpt_caption:
            prompt = captioner(image)
        else:
            return error_image, "Failed: This gradio is not launched with gpt-caption support :("

    image_tensor = to_tensor(image).unsqueeze(0)
    try:
        pil_image = model.enhance(
            lq=image_tensor,
            prompt=prompt,
            upscale=upscale,
            return_type="pil",
        )[0]
    except Exception as e:
        return error_image, f"Failed: {e} :("

    return pil_image, f"Success! :)\nUsed prompt: {prompt}"


MARKDOWN = """
## HYPIR: Harnessing Diffusion-Yielded Score Priors for Image Restoration

[GitHub](https://github.com/XPixelGroup/HYPIR) | [Paper](TODO) | [Project Page](TODO)

If HYPIR is helpful for you, please help star the GitHub Repo. Thanks!
"""

block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown(MARKDOWN)
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil")
            prompt = gr.Textbox(label=(
                "Prompt (Input 'auto' to use gpt-generated caption)"
                if args.gpt_caption else "Prompt"
            ))
            upscale = gr.Slider(minimum=1, maximum=8, value=1, label="Upscale Factor", step=1)
            seed = gr.Number(label="Seed", value=-1)
            run = gr.Button(value="Run")
        with gr.Column():
            result = gr.Image(type="pil", format="png")
            status = gr.Textbox(label="status", interactive=False)
        run.click(
            fn=process,
            inputs=[image, prompt, upscale, seed],
            outputs=[result, status],
        )
block.launch(server_name="0.0.0.0" if not args.local else "127.0.0.1", server_port=args.port)
