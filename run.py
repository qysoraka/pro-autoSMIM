""" Wrapper to train and test a medical image segmentation model. """
import argparse
import os
from argparse import Namespace
import torch
import nni
import yaml
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger, CSVLogger
from lightning.fabric.utilities.seed import seed_everything

import wandb
from utils.train_utils import get_rank
from workers.test_net import test_aug_worker, test_worker
from workers.train_net import train_aug_worker, train_worker

torch.set_float32_matmul_precision("high")


def main():
    parser = argparse.ArgumentParser(description="2D Medical Image Segmentation")

    parser.add_argument(
        "--cfg",
        default="./configs.yaml",
        type=str,
        help="Config file used for experiment",
    )
    parser.add_argument(
        "--workers",
        default=10,
        type=int,
        metavar="N",
        help="number of data loading workers",
    )
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")

    # training configuration
    parser.add_argument(
        "-b",
        "--batch_size",
        default=128,
        type=int,
        metavar="N",
        help="mini-batch size (default: 128), this is the total "
        "batch size of all GPUs on the current node when "
        "using Data Parallel or Distributed Data Parallel",
    )
    parser.add_argument(
        "--test_batch_size",
        default=12,
        type=int,
        metavar="N",
        help="inference mini-batch size (default: 1)",
    )
    parser.add_argument(
        "--epoch",
        default=100,
        type=int,
        metavar="N",
        help="training epoch (default: 100)",
    )
    parser.add_argument(
        "--resume",
        default=-1,
        type=int,
        metavar="N",
        help="resume from which fold (default: -1)",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--optimizer",
        default="ADAM",
        choices=("SGD", "ADAM", "RMSprop"),
        help="optimizer to use (SGD | ADAM | RMSprop)",
    )
    parser.add_argument("--pretrained", default="", help="pretrained model weights")
    parser.add_argument("--size", type=int, default=512, help="size of input image")
    parser.add_argument(
        "--dice_loss", action="store_true", help="using dice loss or not"
    )

    # model specifications
    parser.add_argument("--model", type=str, default="UNet", help="model name")
    parser.add_argument(
        "--encoder", type=str, default="resnet50", help="encoder name of the model"
    )
    parser.add_argument(
        "--no_crop", action="store_true", help="disable random resized cropping"
    )

    # experiment conf