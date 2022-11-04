from typing import Dict, List, Tuple, Type
from keras.layers import Layer, Input

from layers.WeightedAdd import WeightedAdd

class BranchOutLayer(Layer):
    def __init__(self, layer_class: Type[Layer], layer_args: Dict) -> None:
        super(BranchOutLayer, self).__init__()
        self.branch_layer = layer_class(**layer_args)
    
    def call(self, inputs):
        return self.branch_layer(inputs)

class BranchInLayer(Layer):
    def __init__(self, input_shape: Tuple, layer_class: Type[Layer], layer_args: Dict) -> None:
        super(BranchInLayer, self).__init__()
        self.input_layer = Input(shape=input_shape)
        self.branch_layer = layer_class(**layer_args)
    
    def call(self,inputs):
        return WeightedAdd([inputs, self.branch_layer(self.input_layer)])