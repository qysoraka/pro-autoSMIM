
import glob
import math
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.segmentation import slic
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
