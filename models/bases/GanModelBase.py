from keras.layers import Conv2D, Dense
from models.bases.ModelBase import ModelBase


class GanModelBase(ModelBase):
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
