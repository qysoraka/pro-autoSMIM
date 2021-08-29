import os
from collections import OrderedDict
from importlib import import_module
import imageio
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from sklearn.metrics import accuracy_score
from medpy.metric import dc, jc
from torch.optim.lr_scheduler import CosineAnnealingLR


from utils.train_utils import jigsaw, make_optimizer, rotate_images


class Base_Module(LightningModule):
    def __init__(self, args):
        super(Base_Module, self).__init__()

        self.args = args

    def init_weights(self, pretrained):
        if os.path.isfile(pretrained):
            print("=> loading pretrained model {}".format(pretrained))
            pretrained_dict = torch.load(pretrained, map_location='cpu')
            pretrained_dict = pretrained_dict["state_dict"]
            model_dic