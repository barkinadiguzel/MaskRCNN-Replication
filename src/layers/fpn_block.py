import torch
import torch.nn as nn
import torch.nn.functional as F

class FPNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lateral_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.smooth_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x, prev_feature=None):
        lateral = self.lateral_conv(x)
        if prev_feature is not None:
            lateral = lateral + F.interpolate(prev_feature, size=lateral.shape[-2:], mode='nearest')
        return self.smooth_conv(lateral)
