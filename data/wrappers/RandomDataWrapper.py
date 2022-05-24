from typing import Tuple
import numpy as np
import tensorflow as tf

class RandomDataWrapper(object):
    def __init__(self, shape: Tuple, mean: np.float32, std: np.float32, samples: int = 1000):
        self.shape = shape
        self.mean = mean
        self.std = std
        self.samples = samples

    def get_dataset(self):
        randoms = np.random.normal(self.mean, self.std,size=(self.samples,*self.shape))
        return tf.data.Dataset.from_tensor_slices((randoms,np.ones_like(randoms)))
    
    def get_single(self):
        return np.random.normal(self.mean, self.std,size=(1,*self.shape))
