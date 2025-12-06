from layers.roi_align import RoIAlignLayer

class RoIProcessor:
    def __init__(self, output_size=(7,7)):
        self.output_size = output_size

    def __call__(self, feature_map, rois):
        return RoIAlignLayer(feature_map, rois, self.output_size)
