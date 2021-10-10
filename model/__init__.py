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
            model_dict = self.state_dict()
            available_pretrained_dict = {}

            for k, v in pretrained_dict.items():
                # print('Pretrained dict: ', k)
                if k in model_dict.keys():
                    if pretrained_dict[k].shape == model_dict[k].shape:
                        available_pretrained_dict[k] = v
                if k[7:] in model_dict.keys():
                    if pretrained_dict[k].shape == model_dict[k[7:]].shape:
                        available_pretrained_dict[k[7:]] = v

            for k, _ in available_pretrained_dict.items():
                print("loading {}".format(k))
            model_dict.update(available_pretrained_dict)
            self.load_state_dict(model_dict, strict=True)

    def load_weights(self, path):
        if os.path.isfile(path):
            print("=> Loading model from {}".format(path))
            checkpoint = torch.load(path, map_location='cpu')
            state_dict = checkpoint["state_dict"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if "module." in k:
                    name = k[7:]  # remove `module.`
                else:
                    name = k
                new_state_dict[name] = v
            self.load_state_dict(new_state_dict)
            print("=> trained model loaded")

    def initialize(self, decoder, pred):
        if decoder is not None:
            for m in decoder.modules():

                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_uniform_(
                        m.weight, mode="fan_in", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        for m in pred.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def configure_optimizers(self):
        optim, kwargs_optimizer = make_optimizer(args=self.args)
        opt = optim(self.parameters(), **kwargs_optimizer)
        scheduler = CosineAnnealingLR(opt, T_max=self.args.epoch)

        return [opt], [scheduler]


class Model(Base_Module):
    def __init__(self, args, criteria):
        super(Model, self).__init__(args=args)

        self.criteria = criteria

    def forward(self, x):
        x0 = self.encoder(x)
        x1 = self.decoder(*x0)
        y = self.pred_seg(x1)

        return y

    def training_step(self, batch, batch_idx):
        input, target = batch
        output = self(input)
        if not self.args.dice_loss:
            loss = self.criteria[0](output, target)
        else:
            loss = self.criteria[0](output, target) + 1.5 * self.criteria[1](output, target)
        self.log(
            "Train Loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )
        return loss

    # def on_train_batch_end(self, outputs, batch, batch_idx, unused=None):
    #     for name, param in self.named_parameters():
    #         if param.grad is None:
    #             print(name)

    def validation_step(self, batch, batch_idx):
        input, target, name = batch

        output = self(input)
        if not self.args.dice_loss:
            loss = self.criteria[0](output, target)
        else:
            loss = self.criteria[0](output, target) + 1.5 * self.criteria[1](output, target)
        dsc = 0
        for j in range(output.size(0)):
            output_temp = torch.argmax(output[j], dim=0).cpu().numpy()
            target_tem