from typing import List, Dict
import os
import torch
from accelerate.logging import get_logger
from peft import LoraConfig
from diffusers import DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from HYPIR.trainer.base import BaseTrainer


logger = get_logger(__name__, log_level="INFO")


class SD2Trainer(BaseTrainer):

    def init_scheduler(self):
        self.scheduler = DDPMScheduler.from_pretrained(self.config.base_model_path, subfolder="scheduler")

    def init_text_models(self):
        self.tokenizer = CLIPTokenizer.from_pretrained(self.config.base_model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.config.base_model_path, subfolder="text_encoder", torch_dtype=self.weight_dtype).to(self.device)
        self.text_encoder.eval().requires_grad_(False)

    def encode_prompt(self, prompt: List[str]) -> Dict[str, torch.Tensor]:
        txt_ids = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
        text_embed = self.text_encoder(txt_ids.to(self.accelerator.device))[0]
        return {"text_embed": text_embed}

    def init_generator(self):
        self.G = UNet2DConditionModel.from_pretrained(
            self.config.base_model_path, subfolder="unet", torch_dtype=self.weight_dtype).to(self.device)
        self.G.eval().requires_grad_(False)

        if self.config.gradient_checkpointing:
            self.G.enable_gradient_checkpointing()

        # Handle lora configuration
        target_modules = self.config.lora_modules
        logger.info(f"Add lora parameters to {target_modules}")
        G_lora_cfg = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_rank,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        self.G.add_adapter(G_lora_cfg)
        lora_params = list(filter(lambda p: p.requires_grad, self.G.parameters()))
        assert lora_params, "Failed to find lora parameters"
        for p in lora_params:
            p.data = p.to(torch.float32)

    def attach_accelerator_hooks(self):
        def save_model_hook(models, weights, output_dir):
            if self.accelerator.is_main_process:
                model = models[0]
                weights.pop(0)
                model = self.unwrap_model(model)
                assert isinstance(model, UNet2DConditionModel)
                state_dict = {}
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        state_dict[name] = param.detach().clone().data
                torch.save(state_dict, os.path.join(output_dir, "state_dict.pth"))

        def load_model_hook(models, input_dir):
            model = models.pop(0)
            assert isinstance(model, UNet2DConditionModel)
            state_dict = torch.load(os.path.join(input_dir, "state_dict.pth"))
            m, u = model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loading lora parameters, unexpected keys: {u}")

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

    def forward_generator(self):
        z_in = self.batch_inputs.z_lq * self.vae.config.scaling_factor
        eps = self.G(
            z_in,
            self.batch_inputs.timesteps,
            encoder_hidden_states=self.batch_inputs.c_txt["text_embed"],
        ).sample
        z = self.scheduler.step(eps, self.config.coeff_t, z_in).pred_original_sample
        x = self.vae.decode(z.to(self.weight_dtype) / self.vae.config.scaling_factor).sample.float()
        return x
