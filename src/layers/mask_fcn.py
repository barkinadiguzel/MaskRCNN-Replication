import torch
import torch.nn as nn

class MaskFCN(nn.Module):
    def __init__(self, in_channels, out_channels=1, num_convs=4):
        super().__init__()
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, in_channels, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        self.conv_block = nn.Sequential(*layers)
        self.out_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.conv_block(x)
        return self.out_conv(x)
