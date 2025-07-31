import random
import os

import gradio as gr
import torchvision.transforms as transforms
from accelerate.utils import set_seed
from PIL import Image

from HYPIR.enhancer.sd2 import SD2Enhancer


work_dir = "/home/xlab-app-center"

# Download weight
model_dir = os.path.join(work_dir, "HYPIR-openxlab-model")
print(f"Download model to {model_dir}")
os.system(f"git clone https://code.openxlab.org.cn/linxinqi/HYPIR.git {model_dir}")
os.system(f"cd {model_dir} && git lfs pull")
print("Done")

error_image = Image.open(os.path.join("assets", "gradio_error_img.png"))
max_size = os.getenv("HYPIR_APP_MAX_SIZE")
if max_size is not None:
    max_size = tuple(int(x) for x in max_size.split(","))
    if len(max_size) != 2:
        raise ValueError(f"Invalid max size: {max_size}")
    print(f"Max size set to {max_size}, max pixels: {max_size[0] * max_size[1]}")
device = os.getenv("HYPIR_APP_DEVICE")
to_tensor = transforms.ToTensor()

model = SD2Enhancer(
    base_model_path="stabilityai/stable-diffusion-2-1-base",
    weight_path=os.path.join(model_dir, "HYPIR_sd2.pth"),
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
    device=device,
)
print("Start to load model")
model.init_models()
print("Done")


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
            prompt = gr.Textbox(label="Prompt")
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
block.launch()
