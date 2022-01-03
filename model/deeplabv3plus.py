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
        )

        self.decoder = DeepLabV3PlusDecoder(
            encoder_channels=self.encoder.out_channels,
            out_channels=256,
            atrous_rates=(12, 24, 36),
            output_stride=16,
        )

        self.pred_seg = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1),
            nn.UpsamplingBiline