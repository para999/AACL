from torch import nn
import models.common as common
from models.AACL_DAT.DAT import DAT
from models.moco import MoCo


class BaseEncoder(nn.Module):
    def __init__(self):
        super(BaseEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            common.ResBlock(common.default_conv, 64, kernel_size=3),
            common.ResBlock(common.default_conv, 64, kernel_size=3),
            common.ResBlock(common.default_conv, 64, kernel_size=3),
            common.ResBlock(common.default_conv, 64, kernel_size=3),
            common.ResBlock(common.default_conv, 64, kernel_size=3),
            common.ResBlock(common.default_conv, 64, kernel_size=3),
            common.ResBlock(common.default_conv, 64, kernel_size=3),
            common.ResBlock(common.default_conv, 64, kernel_size=3),
            common.ResBlock(common.default_conv, 64, kernel_size=3),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
        )

    def forward(self, x):
        feature = self.encoder(x).squeeze(-1).squeeze(-1)
        out = self.mlp(feature)

        return feature, out


class AACL_DAT(nn.Module):
    def __init__(self, base_encoder=BaseEncoder):
        super().__init__()
        # Generator
        self.G = DAT(
            upscale=4,
            in_chans=3,
            img_size=64,
            img_range=1.,
            depth=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            expansion_factor=4,
            resi_connection='1conv',
            split_size=[8, 32],
        )
        # Encoder
        self.M = MoCo(base_encoder=base_encoder)

    def forward(self, x):
        if self.training:
            x_query = x[:, 0, ...]  # b, n, c, h, w
            x_key = x[:, 1, ...]  # b, n, c, h, w

            dp, logits, labels = self.M(x_query, x_key)

            sr = self.G(x_query, dp)

            return sr, logits, labels
        else:
            dp = self.M(x, x)

            # SR
            sr = self.G(x, dp)

            return sr
