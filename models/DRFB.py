import torch
from einops import rearrange
from torch import nn

from model import common


class DRF(nn.Module):  # Degradation Representation Fusion Conv
    def __init__(self, n_feat=64):
        super().__init__()
        self.n_feat = n_feat

        # 修改kernel部分以适应新的输入维度
        self.kernel = nn.Sequential(
            nn.Linear(256, n_feat, bias=False),
            nn.LeakyReLU(0.1, True),  # 可选，根据需要添加激活函数
        )

    def forward(self, lr, dp):
        """
        Input: lr: (B, n_feat, H, W), dp: (B, 256)
        Output: lr: (B, n_feat, H, W)
        """
        B, C, H, W = lr.shape
        dp = self.kernel(dp).view(-1, C, 1, 1)
        shortcut = lr * dp
        lr = shortcut + lr
        return lr


class DRFB(nn.Module):  # Degradation Representation Fusion Block
    def __init__(self, conv=common.default_conv, n_feat=64, kernel_size=3):
        super().__init__()

        self.drf1 = DRF(n_feat)
        self.drf2 = DRF(n_feat)
        self.conv1 = conv(n_feat, n_feat, kernel_size)
        self.conv2 = conv(n_feat, n_feat, kernel_size)

        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, lr, dp):
        '''
        Input: lr: feature map: B * n_feat * H * W, dp: degradation representation: B * n_feat
        Output: lr: (B, n_feat, H, W)
        '''
        out = self.relu(self.drf1(lr, dp))
        out = self.relu(self.conv1(out))
        out = self.relu(self.drf2(out, dp))
        out = self.conv2(out) + lr
        return out


class DRF_T(nn.Module):  # Degradation Representation Fusion Conv (Transformer version)
    def __init__(self, n_feat=180):
        super().__init__()
        self.n_feat = n_feat

        self.kernel = nn.Sequential(
            nn.Linear(256, n_feat, bias=False))

    def forward(self, lr, dp):
        """
        Input: lr: (B, H*W, n_feat), ldp: (B, 256)
        Output: lr: (B, H*W, n_feat)
        """
        B, N, C = lr.shape
        dp = self.kernel(dp).view(-1, 1, C)
        shortcut = lr * dp
        lr = shortcut + lr
        return lr


if __name__ == '__main__':
    upscale = 4
    height = 64
    width = 64
    model = DRFB().cuda()
    print(height, width)

    dp = torch.randn((4, 256)).cuda()
    x = torch.randn((4, 64, height, width)).cuda()
    x1 = torch.randn((4, height * width, 64)).cuda()
    print(dp.shape)
    print(x1.shape)
    y = model(x, dp)

    print(y.shape)
# B, C, H, W = lr.shape
# x = rearrange(lr, "b c h w -> b (h w) c", h=H, w=W).contiguous()
# x = rearrange(x, "b (h w) c -> b c h w")
# for Transformer-based
