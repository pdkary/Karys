

from dataclasses import dataclass, field
from typing import Callable, Tuple

import numpy as np

def map_to_range(input_arr,new_max,new_min):
    img_max = float(np.max(input_arr))
    img_min = float(np.min(input_arr))
    old_range = float(img_max - img_min + 1e-6)
    new_range = (new_max - new_min)
    return new_range*(input_arr - img_min)/old_range + float(new_min)

def map_to_0_255(input_arr): 
    return map_to_range(input_arr,255.0,0.0)

def map_to_n1_1(input_arr): 
    return map_to_range(input_arr,1.0,-1.0)

def map_to_0_1(input_arr): 
    return map_to_range(input_arr,1.0,0.0)

@dataclass
class ImageDataConfig(object):
    image_shape: Tuple
    batch_size: int
    num_batches: int
    flip_lr: bool = True
    train_test_ratio: float = 0.7
    preview_rows: int = 3
    preview_margin: int = 12
    load_scale_func: Callable = field(default_factory=lambda : map_to_0_1)
    save_scale_func: Callable = field(default_factory=lambda : map_to_0_255)

    def __post_init__(self):
        assert self.batch_size % self.preview_rows == 0
        self.preview_cols = self.batch_size//self.preview_rows
