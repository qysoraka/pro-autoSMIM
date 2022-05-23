from multiprocessing.pool import Pool

from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union

import torch
import numpy as np
from torch import einsum
from torch import Tensor
from functools import partial
from scipy.ndimage import distance_transform_edt as distance
from scipy.spatial.distance import directed_hausdorff

EPS = 1e-7


# Assert utils
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Ten