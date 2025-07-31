import logging
import os
import shutil
from pathlib import Path
from typing import overload, List, Dict
import importlib
import warnings
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.serialization import get_unsafe_globals_in_checkpoint, add_safe_globals
from torchvision.utils import make_grid
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm
import transformers
import lpips
import diffusers
from diffusers import AutoencoderKL
from PIL import Image

from HYPIR.model.D import ImageConvNextDiscriminator
from HYPIR.utils.common import instantiate_from_config, log_txt_as_img, print_vram_state, SuppressLogging
from HYPIR.utils.ema import EMAModel
from HYPIR.utils.tabulate import tabulate


logger = get_logger(__name__, log_level="INFO")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


class BatchInput:

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise ValueError(f"Duplicated key in BatchInput: {name}")
        self.__dict__[name] = value

    def update(self, **kwargs):
        for name, value in kwargs.items():
            self.__dict__[name] = value


class BaseTrainer:

    def __init__(self, config):
        self.config = config
        set_seed(config.seed)
        self.init_environment()
        self.init_models()
        self.summary_models()
        self.init_optimizers()
        self.init_dataset()
        self.prepare_all()

    def init_environment(self):
        logging_dir = Path(self.config.output_dir, self.config.logging_dir)
        accelerator_project_config = ProjectConfiguration(project_dir=self.config.output_dir, logging_dir=logging_dir)
        accelerator = Accelerator(
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            log_with=self.config.report_to,
            project_config=accelerator_project_config,
            mixed_precision=self.config.mixed_precision,
        )
        logger.info(accelerator.state, main_process_only=True)
        if accelerator.is_main_process:
            accelerator.init_trackers("train")
        if accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_warning()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()
        if accelerator.is_main_process:
            if self.config.output_dir is not None:
                os.makedirs(self.config.output_dir, exist_ok=True)
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        self.accelerator = accelerator
        self.weight_dtype = weight_dtype
        self.device = accelerator.device

    def unwrap_model(self, model):
        model = self.accelerator.unwrap_model(model)
        return model

    def init_models(self):
        self.init_scheduler()
        self.init_text_models()
        self.init_vae()
        self.init_generator()
        self.init_discriminator()
        self.init_lpips()

    @overload
    def init_scheduler(self):
        ...

    @overload
    def init_text_models(self):
        ...

    @overload
    def encode_prompt(self, prompt: List[str]) -> Dict[str, torch.Tensor]:
        ...

    def init_vae(self):
        self.vae = AutoencoderKL.from_pretrained(
            self.config.base_model_path, subfolder="vae", torch_dtype=self.weight_dtype).to(self.device)
        self.vae.eval().requires_grad_(False)

    def init_lpips(self):
        with warnings.catch_warnings():
            # Suppress warnings from lpips
            warnings.simplefilter("ignore")
            self.net_lpips = lpips.LPIPS(net="vgg", verbose=False).to(self.device)
        self.net_lpips.eval().requires_grad_(False)

    @overload
    def init_generator(self):
        ...

    def init_discriminator(self):
        # Suppress logs from open-clip
        ctx = (
            nullcontext()
            if self.accelerator.is_local_main_process
            else SuppressLogging(logging.WARNING)
        )
        with ctx:
            self.D = ImageConvNextDiscriminator(precision="bf16").to(device=self.device)
        self.D.train().requires_grad_(True)

    def summary_models(self):
        table_data = []
        for attr, value in self.__dict__.items():
            if not isinstance(value, torch.nn.Module):
                continue
            model = value
            model_type = type(model).__name__
            total_params = sum(p.numel() for p in model.parameters()) / 1_000_000
            learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000
            table_data.append([attr, model_type, f"{total_params:.2f}", f"{learnable_params:.2f}"])
        headers = ["Model Name", "Model Type", "Total Parameters (M)", "Learnable Parameters (M)"]
        table = tabulate(table_data, headers=headers, tablefmt="pretty")
        logger.info(f"Model Summary:\n{table}")

    def init_optimizers(self):
        logger.info(f"Creating {self.config.optimizer_type} optimizers")
        if self.config.optimizer_type == "adam":
            optimizer_cls = torch.optim.AdamW
        elif self.config.optimizer_type == "rmsprop":
            optimizer_cls = torch.optim.RMSprop
        else:
            optimizer_cls = None

        self.G_params = list(filter(lambda p: p.requires_grad, self.G.parameters()))
        self.G_opt = optimizer_cls(
            self.G_params,
            lr=self.config.lr_G,
            **self.config.opt_kwargs,
        )

        self.D_params = list(filter(lambda p: p.requires_grad, self.D.parameters()))
        self.D_opt = optimizer_cls(
            self.D_params,
            lr=self.config.lr_D,
            **self.config.opt_kwargs,
        )

    def init_dataset(self):
        data_cfg = self.config.data_config
        dataset = instantiate_from_config(data_cfg.train.dataset)
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=True,
            batch_size=data_cfg.train.batch_size,
            num_workers=data_cfg.train.dataloader_num_workers,
        )
        self.batch_transform = instantiate_from_config(data_cfg.train.batch_transform)

    def prepare_all(self):
        logger.info("Wrapping models, optimizers and dataloaders")
        attrs = ["G", "D", "G_opt", "D_opt", "dataloader"]
        prepared_objs = self.accelerator.prepare(*[getattr(self, attr) for attr in attrs])
        for attr, obj in zip(attrs, prepared_objs):
            setattr(self, attr, obj)
        print_vram_state("After accelerator.prepare", logger=logger)

    def force_optimizer_ckpt_safe(self, checkpoint_dir):
        def get_symbol(s):
            module_name, symbol_name = s.rsplit('.', 1)
            module = importlib.import_module(module_name)
            symbol = getattr(module, symbol_name)
            return symbol

        for file_name in os.listdir(checkpoint_dir):
            if "optimizer" in file_name and not file_name.endswith("safetensors"):
                path = os.path.join(checkpoint_dir, file_name)
                unsafe_globals = get_unsafe_globals_in_checkpoint(path)
                logger.info(f"Unsafe globals in {path}: {unsafe_globals}")
                unsafe_globals = list(map(get_symbol, unsafe_globals))
                add_safe_globals(unsafe_globals)

    def attach_accelerator_hooks(self):
        ...

    def on_training_start(self):
        # Build ema state dict
        logger.info(f"Creating EMA handler, Use EMA = {self.config.use_ema}, EMA decay = {self.config.ema_decay}")
        if self.config.resume_from_checkpoint is not None and self.config.resume_ema:
            ema_resume_pth = os.path.join(self.config.resume_from_checkpoint, "ema_state_dict.pth")
        else:
            ema_resume_pth = None
        self.ema_handler = EMAModel(
            self.unwrap_model(self.G),
            decay=self.config.ema_decay,
            use_ema=self.config.use_ema,
            ema_resume_pth=ema_resume_pth,
            verbose=self.accelerator.is_local_main_process,
        )

        global_step = 0
        if self.config.resume_from_checkpoint:
            path = self.config.resume_from_checkpoint
            ckpt_name = os.path.basename(path)
            logger.info(f"Resuming from checkpoint {path}")
            self.force_optimizer_ckpt_safe(path)
            self.accelerator.load_state(path)
            global_step = int(ckpt_name.split("-")[1])
            init_global_step = global_step
        else:
            init_global_step = 0

        self.global_step = global_step
        self.pbar = tqdm(
            range(0, self.config.max_train_steps),
            initial=init_global_step,
            desc="Steps",
            disable=not self.accelerator.is_main_process,
        )

    def prepare_batch_inputs(self, batch):
        batch = self.batch_transform(batch)
        gt = (batch["GT"] * 2 - 1).float()
        lq = (batch["LQ"] * 2 - 1).float()
        prompt = batch["txt"]
        bs = len(prompt)
        c_txt = self.encode_prompt(prompt)
        z_lq = self.vae.encode(lq.to(self.weight_dtype)).latent_dist.sample()
        timesteps = torch.full((bs,), self.config.model_t, dtype=torch.long, device=self.device)
        self.batch_inputs = BatchInput(
            gt=gt, lq=lq,
            z_lq=z_lq,
            c_txt=c_txt,
            timesteps=timesteps,
            prompt=prompt,
        )

    @overload
    def forward_generator(self) -> torch.Tensor:
        ...

    def optimize_generator(self):
        with self.accelerator.accumulate(self.G):
            self.unwrap_model(self.D).eval().requires_grad_(False)
            x = self.forward_generator()
            self.G_pred = x
            loss_l2 = F.mse_loss(x, self.batch_inputs.gt, reduction="mean") * self.config.lambda_l2
            loss_lpips = self.net_lpips(x, self.batch_inputs.gt).mean() * self.config.lambda_lpips
            loss_disc = self.D(x, for_G=True).mean() * self.config.lambda_gan
            loss_G = loss_l2 + loss_lpips + loss_disc
            self.accelerator.backward(loss_G)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.G_params, self.config.max_grad_norm)
            self.G_opt.step()
            self.G_opt.zero_grad()
        # Log something
        loss_dict = dict(G_total=loss_G, G_mse=loss_l2, G_lpips=loss_lpips, G_disc=loss_disc)
        return loss_dict

    def optimize_discriminator(self):
        gt = self.batch_inputs.gt
        with torch.no_grad():
            x = self.forward_generator()
        self.G_pred = x
        with self.accelerator.accumulate(self.D):
            self.unwrap_model(self.D).train().requires_grad_(True)
            loss_D_real, real_logits = self.D(gt, for_real=True, return_logits=True)
            loss_D_fake, fake_logits = self.D(x, for_real=False, return_logits=True)
            loss_D = loss_D_real.mean() + loss_D_fake.mean()
            self.accelerator.backward(loss_D)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.D_params, self.config.max_grad_norm)
            self.D_opt.step()
            self.D_opt.zero_grad()
        loss_dict = dict(D=loss_D)
        # logits = D(x) w/o sigmoid = log(p_real(x) / p_fake(x))
        with torch.no_grad():
            real_logits = torch.tensor([logit_map.mean() for logit_map in real_logits], device=self.device).mean()
            fake_logits = torch.tensor([logit_map.mean() for logit_map in fake_logits], device=self.device).mean()
        loss_dict.update(dict(D_logits_real=real_logits, D_logits_fake=fake_logits))
        return loss_dict

    def run(self):
        self.attach_accelerator_hooks()
        self.on_training_start()
        self.batch_count = 0
        while self.global_step < self.config.max_train_steps:
            train_loss = {}
            for batch in self.dataloader:
                self.prepare_batch_inputs(batch)
                bs = len(self.batch_inputs.lq)
                generator_step = ((self.batch_count // self.config.gradient_accumulation_steps) % 2) == 0
                if generator_step:
                    loss_dict = self.optimize_generator()
                else:
                    loss_dict = self.optimize_discriminator()

                for k, v in loss_dict.items():
                    avg_loss = self.accelerator.gather(v.repeat(bs)).mean()
                    if k not in train_loss:
                        train_loss[k] = 0
                    train_loss[k] += avg_loss.item() / self.config.gradient_accumulation_steps

                self.batch_count += 1
                if self.accelerator.sync_gradients:
                    if generator_step:
                        # update EMA
                        self.ema_handler.update()
                    state = "Generator     Step" if not generator_step else "Discriminator Step"
                    _, _, peak = print_vram_state(None)
                    self.pbar.set_description(f"{state}, VRAM peak: {peak:.2f} GB")

                if self.accelerator.sync_gradients and not generator_step:
                    self.global_step += 1
                    self.pbar.update(1)
                    log_dict = {}
                    for k in train_loss.keys():
                        log_dict[f"loss/{k}"] = train_loss[k]
                    train_loss = {}
                    self.accelerator.log(log_dict, step=self.global_step)
                    if self.global_step % self.config.log_image_steps == 0 or self.global_step == 1:
                        self.log_images()
                    if self.global_step % self.config.log_grad_steps == 0 or self.global_step == 1:
                        self.log_grads()
                    if self.global_step % self.config.checkpointing_steps == 0 or self.global_step == 1:
                        self.save_checkpoint()

                if self.global_step >= self.config.max_train_steps:
                    break
        self.accelerator.end_training()

    def log_images(self):
        N = 4
        image_logs = dict(
            lq=(self.batch_inputs.lq[:N] + 1) / 2,
            gt=(self.batch_inputs.gt[:N] + 1) / 2,
            G=(self.G_pred[:N] + 1) / 2,
            prompt=(log_txt_as_img((256, 256), self.batch_inputs.prompt[:N]) + 1) / 2,
        )
        if self.config.use_ema:
            # recompute for EMA results
            self.ema_handler.activate_ema_weights()
            with torch.no_grad():
                ema_x = self.forward_generator()
                image_logs["G_ema"] = (ema_x[:N] + 1) / 2
            self.ema_handler.deactivate_ema_weights()

        if not self.accelerator.is_main_process:
            return

        for tracker in self.accelerator.trackers:
            if tracker.name == "tensorboard":
                for tag, images in image_logs.items():
                    tracker.writer.add_image(
                        f"image/{tag}",
                        make_grid(images.float(), nrow=4),
                        self.global_step,
                    )

        for key, images in image_logs.items():
            image_arrs = (images * 255.0).clamp(0, 255).to(torch.uint8) \
                .permute(0, 2, 3, 1).contiguous().cpu().numpy()
            save_dir = os.path.join(
                self.config.output_dir, self.config.logging_dir, "log_images", f"{self.global_step:07}", key)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for i, img in enumerate(image_arrs):
                Image.fromarray(img).save(os.path.join(save_dir, f"sample{i}.png"))

    def log_grads(self):
        self.unwrap_model(self.D).eval().requires_grad_(False)
        x = self.forward_generator()
        loss_l2 = F.mse_loss(x, self.batch_inputs.gt, reduction="mean") * self.config.lambda_l2
        loss_lpips = self.net_lpips(x, self.batch_inputs.gt).mean() * self.config.lambda_lpips
        loss_disc = self.D(x, for_G=True).mean() * self.config.lambda_gan
        losses = [("l2", loss_l2), ("lpips", loss_lpips), ("disc", loss_disc)]
        grad_dict = {}
        self.G_opt.zero_grad()
        for idx, (name, loss) in enumerate(losses):
            retain_graph = idx != len(losses) - 1
            loss.backward(retain_graph=retain_graph)
            lora_module_grads = {}
            for module_name, module in self.unwrap_model(self.G).named_modules():
                for suffix in self.config.log_grad_modules:
                    if module_name.endswith(suffix):
                        flat_grad = torch.cat([
                            p.grad.flatten() for p in module.parameters() if p.requires_grad
                        ])
                        lora_module_grads.setdefault(suffix, []).append(flat_grad)
                        break
            for k, v in lora_module_grads.items():
                grad_dict[f"grad_norm/{k}_{name}"] = torch.norm(torch.cat(v)).item()
            self.G_opt.zero_grad()
        self.accelerator.log(grad_dict, step=self.global_step)

    def save_checkpoint(self):
        if self.accelerator.is_main_process:
            if self.config.checkpoints_total_limit is not None:
                checkpoints = os.listdir(self.config.output_dir)
                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                if len(checkpoints) >= self.config.checkpoints_total_limit:
                    num_to_remove = len(checkpoints) - self.config.checkpoints_total_limit + 1
                    removing_checkpoints = checkpoints[0:num_to_remove]
                    logger.info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")
                    for removing_checkpoint in removing_checkpoints:
                        removing_checkpoint = os.path.join(self.config.output_dir, removing_checkpoint)
                        shutil.rmtree(removing_checkpoint)
            save_path = os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}")
            self.accelerator.save_state(save_path)
            logger.info(f"Saved state to {save_path}")

            # Save ema weights
            self.ema_handler.save_ema_weights(save_path)
            logger.info(f"Saved ema weights to {save_path}")
