from abc import ABC
from typing import Union
import tensorflow as tf
from data.configs.ImageDataConfig import ImageDataConfig
from data.wrappers.ImageDataWrapper import ImageDataWrapper
from keras.layers import Conv2D, Dense
from data.wrappers.RandomDataWrapper import RandomDataWrapper

from models.ModelBase import ModelBase


class GanModelBase(ModelBase, ABC):
    def set_dataconfig(self, dataconfig: ImageDataConfig):
        self.dataconfig = dataconfig

    def set_datawrapper(self, datawrapper: Union[ImageDataWrapper, RandomDataWrapper]):
        self.datawrapper = datawrapper
        self.train_dataset = None
        self.test_dataset = None
        self.dataset = None
    
    def get_datawrapper(self):
        return self.datawrapper
    
    @property
    def conv_layers(self):
        return [x for x in self.layers if isinstance(x,Conv2D)]
    
    @property
    def conv_output_shape(self):
        return self.conv_layers[-1].output_shape

    @property
    def dense_layers(self):
        return [x for x in self.layers if isinstance(x,Dense)]
    
    @property
    def dense_output_shape(self):
        return self.dense_layers[-1].output_shape
