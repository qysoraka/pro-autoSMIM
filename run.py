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
        meta