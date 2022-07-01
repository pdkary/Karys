from typing import Tuple

import numpy as np
from data.configs.DataConfig import DataConfig


class RandomDataConfig(DataConfig):
    def __init__(self, 
                 shape: Tuple, 
                 mean: np.float32, 
                 std: np.float32, 
                 samples: int = 1000):
        self.shape = shape
        self.mean = mean
        self.std = std
        self.samples = samples
    
    @property
    def input_shape(self):
        return self.shape
    
    @property
    def output_shape(self):
        return self.shape
    
    def to_json(self):
        return dict(shape=self.shape,
                    mean=self.mean, 
                    std=self.std,
                    samples=self.samples)
    def __str__(self):
        return str(self.to_json())
