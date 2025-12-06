import torch
import torch.nn as nn

class RPNConv(nn.Module):
    def __init__(self, in_channels, mid_channels=256, n_anchors=9):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, mid_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.cls_logits = nn.Conv2d(mid_channels, n_anchors, 1)  # objectness
        self.bbox_pred = nn.Conv2d(mid_channels, n_anchors*4, 1)  # bbox regression

    def forward(self, x):
        t = self.relu(self.conv(x))
        return self.cls_logits(t), self.bbox_pred(t)
