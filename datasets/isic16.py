import random
import glob
import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from datasets.data_utils import analyze_name, random_crop


cv2.setNumThreads(1)


class ISIC16(Dataset):
    def __init__(self, x, y, names, im_transform, label_transform, train=False):
        self.im_transform = im_transform
        self.label_transform = label_transform
        assert len(x) == len(y)
        assert len(x) == len(names)
        self.dataset_size = len(y)
        self.x = x
        self.y = y
        self.names = names
        self.train = train

    def __len__(self):
        if self.train:
            return self.dataset_size * 2
        else:
            return self.dataset_size

    def _get_index(self, idx):
        if self.train:
            return idx % self.dataset_size
        else:
