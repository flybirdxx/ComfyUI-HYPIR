import torch
from torch import nn
from vision_aided_loss.cv_discriminator import BlurPool, spectral_norm
from vision_aided_loss.cv_losses import multilevel_loss

from HYPIR.model.backbone import ImageOpenCLIPConvNext


class MultiLevelDConv(nn.Module):

    def __init__(
        self,
        level=3,
        in_ch1=[384, 768, 1536],
        in_ch2=512,
        out_ch=256,
        num_classes=0,
        activation=nn.LeakyReLU(0.2, inplace=True),
        down=1,
    ):
        super().__init__()
        self.decoder = nn.ModuleList()
        self.level = level
        for i in range(level - 1):
            self.decoder.append(
                nn.Sequential(
                    BlurPool(in_ch1[i], pad_type="zero", stride=1, pad_off=1) if down > 1 else nn.Identity(),
                    spectral_norm(nn.Conv2d(in_ch1[i], out_ch, kernel_size=3, stride=2 if down > 1 else 1, padding=1 if down == 1 else 0)),
                    activation,
                    BlurPool(out_ch, pad_type="zero", stride=1),
                    spectral_norm(nn.Conv2d(out_ch, 1, kernel_size=1, stride=2)),
                )
            )
        self.decoder.append(nn.Sequential(spectral_norm(nn.Linear(in_ch2, out_ch)), activation))
        self.out = spectral_norm(nn.Linear(out_ch, 1))
        self.embed = None
        if num_classes > 0:
            self.embed = nn.Embedding(num_classes, out_ch)

    def forward(self, x, c=None):
        final_pred = []
        for i in range(self.level - 1):
            final_pred.append(self.decoder[i](x[i]).squeeze(1))
        h = self.decoder[-1](x[-1].float())
        out = self.out(h)

        if self.embed is not None:
            out += torch.sum(self.embed(c) * h, 1, keepdim=True)

        final_pred.append(out)
        # final_pred = torch.cat(final_pred, 1)
        return final_pred


class ImageConvNextDiscriminator(nn.Module):

    def __init__(self, precision="fp32"):
        super().__init__()
        self.model = ImageOpenCLIPConvNext(precision=precision)
        self.model.eval().requires_grad_(False)
        self.decoder = MultiLevelDConv(level=4, in_ch1=[384, 768, 1536], in_ch2=1024, out_ch=512, down=2)
        self.loss_fn = multilevel_loss(alpha=0.8)
        self.register_buffer("image_mean", torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32))
        self.register_buffer("image_std", torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32))

    def train(self, mode=True):
        self.decoder.train(mode)
        return self

    def eval(self):
        self.train(False)
        return self

    def requires_grad_(self, requires_grad=True):
        self.decoder.requires_grad_(requires_grad)
        return self

    def forward(self, x, for_real=True, for_G=False, verbose=False, return_logits=False):
        x = x * 0.5 + 0.5
        x = (x - self.image_mean[:, None, None]) / self.image_std[:, None, None]

        features = self.model.encode_image(x, return_pooled_feats=True)
        if verbose:
            for i, f in enumerate(features):
                print(f"{i}-th feature: {f.shape}")

        features = self.decoder(features)
        if verbose:
            for i, f in enumerate(features):
                print(f"{i}-th feature after decoder: {f.shape}")

        if not return_logits:
            return self.loss_fn(features, for_real=for_real, for_G=for_G)
        else:
            return self.loss_fn(features, for_real=for_real, for_G=for_G), features
