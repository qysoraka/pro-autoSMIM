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
        return self.dataset_size

    def __getitem__(self, idx):
        # image
        input = cv2.imread(self.x[idx])[..., ::-1]
        input = cv2.resize(input, (512, 512), interpolation=cv2.INTER_CUBIC)
        # label
        target = cv2.imread(self.y[idx])
        target = cv2.resize(target, (512, 512), interpolation=cv2.INTER_NEAREST)
        target = target[..., 0]
        # name
        name = self.names[idx]

        im = Image.fromarray(np.uint8(input))
        target = Image.fromarray(np.uint8(target)).convert("1")

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

        # import imageio
        # im_np = (im_t.permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255
        # target_np = (target_t.numpy()) * 255
        # imageio.imwrite('./debug/im.png', np.array(im_np).astype(np.uint8))
        # imageio.imwrite('./debug/gt.png', np.array(