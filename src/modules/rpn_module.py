import torch.nn as nn
from layers.rpn_conv import RPNConv

class RPNModule(nn.Module):
    def __init__(self, in_channels, mid_channels=256, n_anchors=9):
        super().__init__()
        self.rpn_conv = RPNConv(in_channels, mid_channels, n_anchors)

    def forward(self, feature_map):
        cls_logits, bbox_pred = self.rpn_conv(feature_map)
        return cls_logits, bbox_pred
