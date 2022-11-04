from abc import ABC
from typing import List, Tuple
from keras.models import Model
from keras.layers import Input

class GraphableModelBlock(Model, ABC):
    def __init__(self, *args, **kwargs):
        super(GraphableModelBlock, self).__init__(*args, **kwargs)
        self.layer_definitions = []
    
    # simple way to call a stack of layers on a tensor
    def call(self, input_tensor, training=False):
        x = input_tensor
        for layer in self.layer_definitions:
            x = layer(x, training = training)
        return x 

    # A convenient way to get model summary 
    # and plot in subclassed api
    def build_graph(self):
        if isinstance(self.input_shape, Tuple):
            x = Input(shape=self.input_shape)
        elif isinstance(self.input_shape, List):
            x = [Input(shape=s) for s in self.input_shape]
        return Model(inputs=x, outputs=self.call(x))
