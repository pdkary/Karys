from typing import Dict, Type
from keras.layers import Conv2D, LeakyReLU, Layer, BatchNormalization, Conv2DTranspose

from karys.models.bases import GraphableModelBlock

class LayerNormActBlock(GraphableModelBlock):
    def __init__(self, 
        n_blocks: int, 
        layer_class: Type[Layer],
        norm_class: Type[Layer] = None, 
        act_class: Type[Layer] = None,  
        layer_args:  Dict = {}, 
        norm_args: Dict = {},
        act_args: Dict = {}):
            super(LayerNormActBlock, self).__init__()
            self._layers = [layer_class(**layer_args) for n in range(n_blocks)]
            self.normalizations = [norm_class(**norm_args)  if norm_class is not None else None for n in range(n_blocks)]
            self.activations = [act_class(**act_args) if act_class is not None else None for n in range(n_blocks)]
    
    def call(self, input_tensor, training=False):
        x = input_tensor
        for layer, norm, act in zip(self._layers, self.normalizations, self.activations):
            x = layer(x, training = training)
            if norm is not None:
                x = norm(x, training = training)
            if act is not None:
                x = act(x, training = training)
        return x

class Conv2DNormActBlock(LayerNormActBlock):
    def __init__(self, 
        n_blocks: int, 
        norm_class: Type[Layer] = None, 
        act_class: Type[Layer] = None,  
        conv_args:  Dict = {}, 
        norm_args: Dict = {},
        act_args: Dict = {}):
            super(Conv2DNormActBlock, self).__init__(
                n_blocks=n_blocks,
                layer_class=Conv2D,
                norm_class=norm_class,
                act_class=act_class,
                layer_args=conv_args,
                norm_args=norm_args,
                act_args=act_args)

class Conv2DLeakyReluBlock(Conv2DNormActBlock):
    def __init__(self, n_blocks: int, alpha: float, conv_args: Dict):
        super(Conv2DLeakyReluBlock, self).__init__(
            n_blocks, 
            act_class=LeakyReLU, 
            act_args=dict(alpha=alpha), 
            conv_args=conv_args)

class Conv2DBatchNormLeakyReluBlock(Conv2DNormActBlock):
    def __init__(self, n_blocks: int, alpha: float, conv_args: Dict):
        super(Conv2DBatchNormLeakyReluBlock, self).__init__(
            n_blocks, 
            norm_class=BatchNormalization, 
            act_class=LeakyReLU, 
            act_args=dict(alpha=alpha), 
            conv_args=conv_args)

class Conv2DTransposeNormActBlock(LayerNormActBlock):
    def __init__(self, 
        n_blocks: int, 
        norm_class: Type[Layer] = None, 
        act_class: Type[Layer] = None,  
        conv_args:  Dict = {}, 
        norm_args: Dict = {},
        act_args: Dict = {}):
            super(Conv2DTransposeNormActBlock, self).__init__(
                n_blocks=n_blocks,
                layer_class=Conv2DTranspose,
                norm_class=norm_class,
                act_class=act_class,
                layer_args=conv_args,
                norm_args=norm_args,
                act_args=act_args)

class Conv2DTransposeLeakyReluBlock(Conv2DTransposeNormActBlock):
    def __init__(self, n_blocks: int, alpha: float, conv_args: Dict):
        super(Conv2DTransposeLeakyReluBlock, self).__init__(
            n_blocks, 
            act_class=LeakyReLU, 
            act_args=dict(alpha=alpha), 
            conv_args=conv_args)

class Conv2DTransposeBatchNormLeakyReluBlock(Conv2DTransposeNormActBlock):
    def __init__(self, n_blocks: int, alpha: float, conv_args: Dict):
        super(Conv2DTransposeBatchNormLeakyReluBlock, self).__init__(
            n_blocks, 
            norm_class=BatchNormalization, 
            act_class=LeakyReLU, 
            act_args=dict(alpha=alpha), 
            conv_args=conv_args)
