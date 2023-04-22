from typing import Dict, Tuple, Type
from keras.layers import Conv2D, LeakyReLU, Layer, BatchNormalization, Conv2DTranspose
from tensorflow_addons.layers import InstanceNormalization
from karys.models.bases import GraphableModelBlock
from karys.layers.AdaptiveInstanceNormalization import AutoAdaptiveInstanceNormalization

class LayerNormActBlock(GraphableModelBlock):
    instance_count = 0
    def __init__(self, 
        n_blocks: int = 1,
        input_shape: Tuple = (None, None, None, None),
        layer_class: Type[Layer] = None,
        norm_class: Type[Layer] = None, 
        act_class: Type[Layer] = None,  
        layer_args:  Dict = {}, 
        norm_args: Dict = {},
        act_args: Dict = {},
        name: str = None):
            name = name if name is not None else f"layer_norm_act_block_{LayerNormActBlock.instance_count}"
            super(LayerNormActBlock, self).__init__(name=name)
            LayerNormActBlock.instance_count += 1
            self.n_blocks = n_blocks
            self.in_shape = input_shape
            self.layer_class = layer_class
            self.norm_class = norm_class
            self.act_class = act_class
            self.layer_args = layer_args
            self.norm_args = norm_args
            self.act_args = act_args
            self._layers = [layer_class(**layer_args) for n in range(n_blocks)]
            self.normalizations = [norm_class(**norm_args)  if norm_class is not None else None for n in range(n_blocks)]
            self.activations = [act_class(**act_args) if act_class is not None else None for n in range(n_blocks)]

            self.layer_definitions = []
            for layer, norm, act in zip(self._layers, self.normalizations, self.activations):
                self.layer_definitions.append(layer)
                if norm is not None:
                    self.layer_definitions.append(norm)
                if act is not None:
                    self.layer_definitions.append(act)
    
    @property
    def input_shape(self):
         return self.in_shape
    
    
    def get_config(self):
         return {
              "n_blocks": self.n_blocks,
              "input_shape": self.in_shape,
              "name": self.name
            }

class Conv2DNormActBlock(LayerNormActBlock):
    instance_count = 0
    def __init__(self, 
        n_blocks: int = 1, 
        input_shape: Tuple = (None, None, None, None),
        norm_class: Type[Layer] = None, 
        act_class: Type[Layer] = None,  
        conv_args:  Dict = {"use_bias": False}, 
        norm_args: Dict = {},
        act_args: Dict = {},
        name: str = None):
            super(Conv2DNormActBlock, self).__init__(
                n_blocks=n_blocks,
                input_shape=input_shape,
                layer_class=Conv2D,
                norm_class=norm_class,
                act_class=act_class,
                layer_args=conv_args,
                norm_args=norm_args,
                act_args=act_args,
                name = name if name is not None else f"conv2d_norm_act_block_{Conv2DNormActBlock.instance_count}")
            Conv2DNormActBlock.instance_count += 1
    
    def get_config(self):
         return super().get_config()
    
class Conv2DLeakyReluBlock(Conv2DNormActBlock):
    instance_count = 0
    def __init__(self, 
                 n_blocks: int = 1, 
                 input_shape: Tuple = (None, None, None, None), 
                 alpha: float = 0.8, 
                 conv_args: Dict = {},
                 name: str = None):
        name_base =  f"conv_2d_leaky_relu_block_{Conv2DLeakyReluBlock.instance_count}"
        super(Conv2DLeakyReluBlock, self).__init__(
            n_blocks=n_blocks,
            input_shape=input_shape,
            act_class=LeakyReLU, 
            act_args=dict(alpha=alpha), 
            conv_args=conv_args,
            name = f"{name}_{name_base}" if name is not None else name_base)
        Conv2DLeakyReluBlock.instance_count += 1
        
    def get_config(self):
         base_config = super().get_config()
         base_config.update({"alpha": self.act_args["alpha"], "conv_args": self.layer_args})
         return base_config

class Conv2DBatchNormLeakyReluBlock(Conv2DNormActBlock):
    instance_count = 0
    def __init__(self, 
                 n_blocks: int = 1, 
                 input_shape: Tuple = (None, None, None, None), 
                 alpha: float = 0.8, 
                 conv_args: Dict = {},
                 name: str = None):
        name_base =  f"conv_2d_batch_norm_leaky_relu_block_{Conv2DBatchNormLeakyReluBlock.instance_count}"
        super(Conv2DBatchNormLeakyReluBlock, self).__init__(
            n_blocks=n_blocks,
            input_shape=input_shape,
            norm_class=BatchNormalization, 
            act_class=LeakyReLU, 
            act_args=dict(alpha=alpha), 
            conv_args=conv_args,
            name = f"{name}_{name_base}" if name is not None else name_base)
        Conv2DBatchNormLeakyReluBlock.instance_count += 1
    
    def get_config(self):
         base_config = super().get_config()
         base_config.update({"alpha": self.act_args["alpha"], "conv_args": self.layer_args})
         return base_config

class Conv2DInstanceNormLeakyReluBlock(Conv2DNormActBlock):
    def __init__(self, 
                 n_blocks: int = 1, 
                 input_shape: Tuple = (None, None, None, None),
                 alpha: float = 0.8, 
                 conv_args: Dict = {},
                 name: str = None):
        name_base =  f"conv_2d_instance_norm_leaky_relu_block_{Conv2DInstanceNormLeakyReluBlock.instance_count}"
        super(Conv2DInstanceNormLeakyReluBlock, self).__init__(
            n_blocks=n_blocks,
            input_shape=input_shape,
            norm_class=InstanceNormalization, 
            act_class=LeakyReLU, 
            act_args=dict(alpha=alpha), 
            conv_args=conv_args,
            name = f"{name}_{name_base}" if name is not None else name_base)
        Conv2DInstanceNormLeakyReluBlock.instance_count += 1
    
    def get_config(self):
         base_config = super().get_config()
         base_config.update({"alpha": self.act_args["alpha"], "conv_args": self.layer_args})
         return base_config

class Conv2DAutoAdaNormLeakyReluBlock(Conv2DNormActBlock):
    def __init__(self, 
                 n_blocks: int = 1, 
                 input_shape: Tuple = (None, None, None, None),
                 alpha: float = 0.8, 
                 conv_args: Dict = {},
                 name: str = None):
        name_base  = f"conv_2d_auto_ada_norm_leaky_relu_block_{Conv2DAutoAdaNormLeakyReluBlock.instance_count}"
        super(Conv2DAutoAdaNormLeakyReluBlock, self).__init__(
            n_blocks=n_blocks,
            input_shape=input_shape,
            norm_class=AutoAdaptiveInstanceNormalization, 
            act_class=LeakyReLU, 
            act_args=dict(alpha=alpha), 
            conv_args=conv_args,
            name = f"{name}_{name_base}" if name is not None else name_base)
        Conv2DAutoAdaNormLeakyReluBlock.instance_count += 1
    
    def get_config(self):
         base_config = super().get_config()
         base_config.update({"alpha": self.act_args["alpha"], "conv_args": self.layer_args})
         return base_config

class Conv2DTransposeNormActBlock(LayerNormActBlock):
    instance_count = 0
    def __init__(self,
        n_blocks: int = 1,
        input_shape: Tuple = (None, None, None, None),
        norm_class: Type[Layer] = None, 
        act_class: Type[Layer] = None,  
        conv_args:  Dict = {}, 
        norm_args: Dict = {},
        act_args: Dict = {},
        name: str = None):
            name_base = f"conv_2d_transpose_norm_act_block_{Conv2DTransposeNormActBlock.instance_count}"
            super(Conv2DTransposeNormActBlock, self).__init__(
                n_blocks=n_blocks,
                input_shape=input_shape,
                layer_class=Conv2DTranspose,
                norm_class=norm_class,
                act_class=act_class,
                layer_args=conv_args,
                norm_args=norm_args,
                act_args=act_args,
                name = f"{name}_{name_base}" if name is not None else name_base)
            Conv2DTransposeNormActBlock.instance_count += 1
    
    def get_config(self):
         base_config = super().get_config()
         base_config.update({"alpha": self.act_args["alpha"], "conv_args": self.layer_args})
         return base_config

class Conv2DTransposeLeakyReluBlock(Conv2DTransposeNormActBlock):
    instance_count = 0
    def __init__(self, 
                 n_blocks: int = 1, 
                 input_shape: Tuple = (None, None, None, None),
                 alpha: float = 0.8, 
                 conv_args: Dict = {},
                 name: str = None):
        name_base = f"conv_2d_transpose_leaky_relu_block_{Conv2DTransposeLeakyReluBlock.instance_count}"
        super(Conv2DTransposeLeakyReluBlock, self).__init__(
            n_blocks=n_blocks,
            input_shape=input_shape,
            act_class=LeakyReLU, 
            act_args=dict(alpha=alpha), 
            conv_args=conv_args,
            name = f"{name}_{name_base}" if name is not None else name_base)
        Conv2DTransposeLeakyReluBlock.instance_count += 1
    
    def get_config(self):
         base_config = super().get_config()
         base_config.update({"alpha": self.act_args["alpha"], "conv_args": self.layer_args})
         return base_config

class Conv2DTransposeBatchNormLeakyReluBlock(Conv2DTransposeNormActBlock):
    def __init__(self, 
                 n_blocks: int = 1, 
                 input_shape: Tuple = (None, None, None, None),
                 alpha: float = 0.8, 
                 conv_args: Dict = {},
                 name: str = None):
        name_base = f"conv_2d_transpose_batch_norm_leaky_relu_block_{Conv2DTransposeBatchNormLeakyReluBlock.instance_count}"
        super(Conv2DTransposeBatchNormLeakyReluBlock, self).__init__(
            n_blocks=n_blocks,
            input_shape=input_shape,
            norm_class=BatchNormalization, 
            act_class=LeakyReLU, 
            act_args=dict(alpha=alpha), 
            conv_args=conv_args,
            name = f"{name}_{name_base}" if name is not None else name_base)
        Conv2DTransposeBatchNormLeakyReluBlock.instance_count += 1
    
    def get_config(self):
         base_config = super().get_config()
         base_config.update({"alpha": self.act_args["alpha"], "conv_args": self.layer_args})
         return base_config

