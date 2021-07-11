import glob
import os
import random
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from datasets.data_utils import analyze_name

cv2.setNumThreads(1)


class ISIC17(Dataset):
    def __init__(self, x, y, names, im_transform, label_t