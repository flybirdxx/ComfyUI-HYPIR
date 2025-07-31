import torch
from torch import nn
import open_clip
from open_clip.factory import CLIP


def _visual_forward(
    model: CLIP,
    image: torch.Tensor,
    return_feats: bool = False,
    return_pooled_feats: bool = False,
):
    # stem, stages, norm_pre
    x, intermediates = model.visual.trunk.forward_intermediates(
        image,
        indices=None,
        norm=False,  # useless
        stop_early=False,
        intermediates_only=False,
    )
    if return_feats:
        return intermediates[1:]
    # trunk.head
    x = model.visual.trunk.forward_head(x)
    # visual.head
    x = model.visual.head(x)
    if return_pooled_feats:
        intermediates[-1] = x
        return intermediates[1:]
    return x


class ImageOpenCLIPConvNext(nn.Module):

    def __init__(self, precision="fp32"):
        super().__init__()
        self.model, _, _ = open_clip.create_model_and_transforms(
            "convnext_xxlarge",
            pretrained="laion2b_s34b_b82k_augreg_soup",
            precision=precision,
        )

    def encode_image(self, image, return_feats=False, return_pooled_feats=False):
        return _visual_forward(
            self.model,
            image,
            return_feats,
            return_pooled_feats,
        )
