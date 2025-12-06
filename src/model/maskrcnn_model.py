import torch
import torch.nn as nn
from torchvision.ops import roi_align
from modules.backbone_resnet import BackboneResNet
from modules.rpn_module import RPNModule
from modules.roi_processor import RoIProcessor
from layers.mask_fcn import MaskFCN
from layers.box_fc import BoxFC

class MaskRCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Backbone
        self.backbone = BackboneResNet()

        # RPN
        self.rpn = RPNModule(in_channels=1024)

        # RoI Align processor
        self.roi_processor = RoIProcessor(output_size=(14,14))

        # Box & Mask heads
        self.box_head = BoxFC(in_features=1024*14*14, num_classes=num_classes)
        self.mask_head = MaskFCN(in_channels=1024, out_channels=num_classes)

    def forward(self, images, rois):
        # Feature extraction
        c2, c3, c4, c5 = self.backbone(images)

        # RPN
        rpn_cls, rpn_bbox = self.rpn(c4)

        # RoI Align
        roi_feats = self.roi_processor(c4, rois)

        # Box head (flatten)
        roi_feats_flat = roi_feats.view(roi_feats.size(0), -1)
        cls_score, bbox_pred = self.box_head(roi_feats_flat)

        # Mask head
        mask_pred = self.mask_head(roi_feats)

        return cls_score, bbox_pred, mask_pred, rpn_cls, rpn_bbox
