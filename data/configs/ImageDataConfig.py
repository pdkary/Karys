

from dataclasses import dataclass, field
from typing import Callable, Tuple

import numpy as np

from data.configs.DataConfig import DataConfig

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
class ImageDataConfig(DataConfig):
    image_shape: Tuple
    image_type: str
    flip_lr: bool = True
    preview_rows: int = 3
    preview_cols: int = 4
    preview_margin: int = 12
    load_n_percent: int = 100
    load_scale_func: Callable = field(default_factory=lambda : map_to_0_1)
    save_scale_func: Callable = field(default_factory=lambda : map_to_0_255)

    @property
    def input_shape(self):
        return self.image_shape
    
    @property
    def output_shape(self):
        return self.image_shape

    def to_json(self):
        return dict(image_shape=self.image_shape,
                    image_type=self.image_type,
                    flip=self.flip_lr,
                    preview_rows=self.preview_rows,
                    preview_cols=self.preview_cols,
                    preview_margin=self.preview_margin,
                    load_n_percent=self.load_n_percent)

    def __str__(self):
        return str(self.to_json())

    def load_from_saved_configs(self):
        pass

    # @classmethod
    # def load_from_saved_configs(cls, filepath, load_scale_func: Callable, save_scale_func: Callable):
    #     json_dict = SavedModelService.get_data_reference_dict()[filepath]
    #     json_dict['load_scale_func'] = load_scale_func
    #     json_dict['save_scale_func'] = save_scale_func
    #     return cls(**json_dict)
