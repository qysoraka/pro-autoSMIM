from torch import nn
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.deeplabv3.decoder import DeepLabV3PlusDecoder
from model import Model, Context_Model, Rotation_Model, Jigsaw_Model


class Deeplabv3plus(Model):
    def __init__(self, args, input_channel=3, criteria=None):
        super(Deeplabv3plus, self).__init__(args=args, criteria=criteria)

        self.encoder = get_encoder(
            name='resnet50',
            in_channels=input_channel,
            depth=5,
            weights='imagenet',
            output_stride=16,
