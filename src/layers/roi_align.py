import torch
from torchvision.ops import roi_align

def RoIAlignLayer(feature_maps, rois, output_size=(7,7)):
    return roi_align(feature_maps, rois, output_size)
