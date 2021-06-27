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
            return idx

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx = self._get_index(idx)

        # BGR -> RGB -> PIL
        _input = cv2.imread(self.x[idx])[..., ::-1]
        _input = cv2.resize(_input, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        # label
        _target = cv2.imread(self.y[idx])
        _target = cv2.resize(_target, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        _target = _target[..., 0]

        name = self.names[idx]
        mask = np.ones_like(_target)

        if self.train:
            image, label = random_crop(_input, _target, roi=mask, size=[0.6, 1.0])
        else:
            image = _input.copy()
            label = _target.copy()

        im = Image.fromarray(np.uint8(image))
        target = Image.fromarray(np.uint8(label)).convert("1")

        # identical transformation for im and gt
        seed = np.random.randint(2147483647)
        torch.manual_seed(seed)
        random.seed(seed)

        if self.im_transform is not None:
            im_t = self.im_transform(im)

        torch.manual_seed(seed)
        random.seed(seed)
        if self.label_transform is not None:
            target_t = self.label_transform(target)
            target_t = torch.squeeze(target_t).long()

        # For debugging
        # import imageio
        # im_np = (im_t.permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255
        # target_np = (target_t.numpy()) * 255
        # imageio.imwrite('./debug/im.png', np.array(im_np).astype(np.uint8))
        # imageio.imwrite('./debug/gt.png', np.array(target_np).astype(np.uint8))

        if self.train:
            return im_t, target_t
        else:
            return im_t, target_t, name


def load_name():

    inputs, targets, names = [], [], []
    test_inputs, test_targets, test_names = [], [], []

    # Need modification
    input_pattern = glob.glob(
        "/data/share/wangzh/datasets/ISIC2016/Training_Data/*.jpg"
  