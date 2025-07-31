import os
from collections import OrderedDict

import torch


class EMAModel:

    def __init__(self, model, decay=0.999, use_ema=False, ema_resume_pth=None, verbose=False):
        self.use_ema = use_ema
        if self.use_ema:
            self.model = model
            self.decay = decay
            self.ema_state_dict = OrderedDict()
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.ema_state_dict[name] = param.clone().detach()
            self.original_weights = None
            if verbose:
                print(f"Keep EMA Parameters: {len(self.ema_state_dict)}")
            if ema_resume_pth:
                if verbose:
                    print(f"Loading EMA Parameters from {ema_resume_pth}")
                ema_ckpt = torch.load(ema_resume_pth, map_location="cpu")
                for name, param in self.ema_state_dict.items():
                    if name in ema_ckpt:
                        _param = ema_ckpt[name].to(param.dtype).to(param.device)
                        self.ema_state_dict[name].copy_(_param.data)

    def update(self):
        if not self.use_ema:
            return
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.ema_state_dict[name] = self.decay * self.ema_state_dict[name] + (1 - self.decay) * param.clone().detach()

    def activate_ema_weights(self):
        if not self.use_ema:
            return
        self.original_weights = OrderedDict()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.original_weights[name] = param.clone().detach()
                param.data.copy_(self.ema_state_dict[name].data)

    def deactivate_ema_weights(self):
        if not self.use_ema:
            return
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.original_weights[name].data)
        self.original_weights = None

    def save_ema_weights(self, save_dir):
        if not self.use_ema:
            return
        save_path = os.path.join(save_dir, "ema_state_dict.pth")
        torch.save(self.ema_state_dict, save_path)
