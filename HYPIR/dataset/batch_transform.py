from typing import Any, overload, Dict, List, Sequence
import random
import copy

import torch
from torch.nn import functional as F
import numpy as np

from HYPIR.dataset.utils import USMSharp, filter2D
from HYPIR.dataset.diffjpeg import DiffJPEG
from HYPIR.utils.degradation import random_add_gaussian_noise_pt, random_add_poisson_noise_pt


class BatchTransform:

    @overload
    def __call__(self, batch: Any) -> Any: ...


class IdentityBatchTransform(BatchTransform):

    def __call__(self, batch: Any) -> Any:
        return batch


class RealESRGANBatchTransform(BatchTransform):

    def __init__(
        self,
        hq_key,
        extra_keys,
        use_sharpener,
        queue_size,
        resize_prob,
        resize_range,
        gray_noise_prob,
        gaussian_noise_prob,
        noise_range,
        poisson_scale_range,
        jpeg_range,
        second_blur_prob,
        stage2_scale,
        resize_prob2,
        resize_range2,
        gray_noise_prob2,
        gaussian_noise_prob2,
        noise_range2,
        poisson_scale_range2,
        jpeg_range2,
        resize_back=True,
    ):
        super().__init__()
        self.hq_key = hq_key
        self.extra_keys = extra_keys

        # resize settings for the first degradation process
        self.resize_prob = resize_prob
        self.resize_range = resize_range

        # noise settings for the first degradation process
        self.gray_noise_prob = gray_noise_prob
        self.gaussian_noise_prob = gaussian_noise_prob
        self.noise_range = noise_range
        self.poisson_scale_range = poisson_scale_range
        self.jpeg_range = jpeg_range

        self.second_blur_prob = second_blur_prob
        self.stage2_scale = stage2_scale

        # resize settings for the second degradation process
        self.resize_prob2 = resize_prob2
        self.resize_range2 = resize_range2

        # noise settings for the second degradation process
        self.gray_noise_prob2 = gray_noise_prob2
        self.gaussian_noise_prob2 = gaussian_noise_prob2
        self.noise_range2 = noise_range2
        self.poisson_scale_range2 = poisson_scale_range2
        self.jpeg_range2 = jpeg_range2

        self.use_sharpener = use_sharpener
        if self.use_sharpener:
            self.usm_sharpener = USMSharp()
        else:
            self.usm_sharpener = None
        self.queue_size = queue_size
        self.jpeger = DiffJPEG(differentiable=False)

        self.queue = {}
        self.resize_back = resize_back

    @torch.no_grad()
    def _dequeue_and_enqueue(self, values: Dict[str, torch.Tensor | List[str]]) -> Dict[str, torch.Tensor | List[str]]:
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        if len(self.queue):
            if set(values.keys()) != set(self.queue.keys()):
                raise ValueError(f"Key mismatch, input keys: {values.keys()}, queue keys: {self.queue.keys()}")
        else:
            for k, v in values.items():
                if not isinstance(v, (torch.Tensor, list)):
                    raise TypeError(f"Queue of type {type(v)} is not supported")
                if isinstance(v, list) and not isinstance(v[0], str):
                    raise TypeError("Only support queue for list of string")

                if isinstance(v, torch.Tensor):
                    size = (self.queue_size, *v.shape[1:])
                    self.queue[k] = torch.zeros(size=size, dtype=v.dtype, device=v.device)
                elif isinstance(v, list):
                    self.queue[k] = [None] * self.queue_size
            self.queue_ptr = 0

        for k, v in values.items():
            if self.queue_size % len(v) != 0:
                raise ValueError(f"Queue size {self.queue_size} should be divisible by batch size {len(v)} for key {k}")

        results = {}
        if self.queue_ptr == self.queue_size:
            # The queue is full, do dequeue and enqueue
            idx = torch.randperm(self.queue_size)
            for k, q in self.queue.items():
                v = values[k]
                b = len(v)
                if isinstance(q, torch.Tensor):
                    # Shuffle the queue
                    q_shuf = q[idx]
                    # Get front samples
                    results[k] = q_shuf[0:b, ...].clone()
                    # Update front samples
                    q_shuf[0:b, ...] = v.clone()
                    self.queue[k] = q_shuf
                else:
                    q_shuf = [q[i] for i in idx]
                    results[k] = q_shuf[0:b]
                    for i in range(b):
                        q_shuf[i] = v[i]
                    self.queue[k] = q_shuf
        else:
            # Only do enqueue
            for k, q in self.queue.items():
                v = values[k]
                b = len(v)
                if isinstance(q, torch.Tensor):
                    q[self.queue_ptr : self.queue_ptr + b, ...] = v.clone()
                else:
                    for i in range(b):
                        q[self.queue_ptr + i] = v[i]
            results = copy.deepcopy(values)
            self.queue_ptr = self.queue_ptr + b

        return results

    @torch.no_grad()
    def __call__(self, batch: Dict[str, torch.Tensor | List[str]]) -> Dict[str, torch.Tensor | List[str]]:
        hq = batch[self.hq_key]
        if self.use_sharpener:
            self.usm_sharpener.to(hq)
            hq = self.usm_sharpener(hq)
        self.jpeger.to(hq)

        kernel1 = batch["kernel1"]
        kernel2 = batch["kernel2"]
        sinc_kernel = batch["sinc_kernel"]

        ori_h, ori_w = hq.size()[2:4]

        # ----------------------- The first degradation process ----------------------- #
        # blur
        out = filter2D(hq, kernel1)
        # random resize
        updown_type = random.choices(["up", "down", "keep"], self.resize_prob)[0]
        if updown_type == "up":
            scale = np.random.uniform(1, self.resize_range[1])
        elif updown_type == "down":
            scale = np.random.uniform(self.resize_range[0], 1)
        else:
            scale = 1
        mode = random.choice(["area", "bilinear", "bicubic"])
        out = F.interpolate(out, scale_factor=scale, mode=mode)
        # add noise
        if np.random.uniform() < self.gaussian_noise_prob:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=self.noise_range,
                clip=True,
                rounds=False,
                gray_prob=self.gray_noise_prob,
            )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.poisson_scale_range,
                gray_prob=self.gray_noise_prob,
                clip=True,
                rounds=False,
            )
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range)
        # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        out = torch.clamp(out, 0, 1)
        out = self.jpeger(out, quality=jpeg_p)

        # ----------------------- The second degradation process ----------------------- #
        # blur
        if np.random.uniform() < self.second_blur_prob:
            out = filter2D(out, kernel2)

        # select scale of second degradation stage
        if isinstance(self.stage2_scale, Sequence):
            min_scale, max_scale = self.stage2_scale
            stage2_scale = np.random.uniform(min_scale, max_scale)
        else:
            stage2_scale = self.stage2_scale
        stage2_h, stage2_w = int(ori_h / stage2_scale), int(ori_w / stage2_scale)
        # print(f"stage2 scale = {stage2_scale}")

        # random resize
        updown_type = random.choices(["up", "down", "keep"], self.resize_prob2)[0]
        if updown_type == "up":
            scale = np.random.uniform(1, self.resize_range2[1])
        elif updown_type == "down":
            scale = np.random.uniform(self.resize_range2[0], 1)
        else:
            scale = 1
        mode = random.choice(["area", "bilinear", "bicubic"])
        out = F.interpolate(out, size=(int(stage2_h * scale), int(stage2_w * scale)), mode=mode)
        # add noise
        if np.random.uniform() < self.gaussian_noise_prob2:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=self.noise_range2,
                clip=True,
                rounds=False,
                gray_prob=self.gray_noise_prob2,
            )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.poisson_scale_range2,
                gray_prob=self.gray_noise_prob2,
                clip=True,
                rounds=False,
            )

        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        if np.random.uniform() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(["area", "bilinear", "bicubic"])
            out = F.interpolate(out, size=(stage2_h, stage2_w), mode=mode)
            out = filter2D(out, sinc_kernel)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range2)
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
        else:
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range2)
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(["area", "bilinear", "bicubic"])
            out = F.interpolate(out, size=(stage2_h, stage2_w), mode=mode)
            out = filter2D(out, sinc_kernel)

        # resize back to gt_size since We are doing restoration task
        if stage2_scale != 1 and self.resize_back:
            out = F.interpolate(out, size=(ori_h, ori_w), mode="bicubic")
        # clamp and round
        lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.0

        batch = {"GT": hq, "LQ": lq, **{k: batch[k] for k in self.extra_keys}}
        if self.queue_size > 0:
            batch = self._dequeue_and_enqueue(batch)
        return batch
