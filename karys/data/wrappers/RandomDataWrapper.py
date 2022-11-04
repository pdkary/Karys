from typing import Tuple
import numpy as np
import tensorflow as tf
from karys.data.configs.RandomDataConfig import RandomDataConfig

from karys.data.wrappers.DataWrapper import DataWrapper

class RandomDataWrapper(DataWrapper):
    def __init__(self, data_config: RandomDataConfig, train_test_ratio: float = 0.7):
        super(RandomDataWrapper, self).__init__(data_config, train_test_ratio)

    def get_dataset(self):
        return np.random.normal(self.data_config.mean, self.data_config.std,size=(self.data_config.samples,*self.data_config.input_shape))
    
    def get_train_dataset(self):
        return self.get_dataset()

    def get_validation_dataset(self):
        return self.get_dataset()
    
    @classmethod
    def load_from_file(cls,filename, DataConfig):
        pass
