from __future__ import annotations
from karys.layers.WeightedAdd import WeightedAdd

from karys.models.bases import GraphableModelBlock
from karys.models.convolutional_blocks import Conv2DAutoAdaNormLeakyReluBlock, Conv2DBatchNormLeakyReluBlock
from keras.layers import UpSampling2D, Dense, Activation,  Reshape
from keras.initializers import Constant

from karys.models.passthrough_blocks import PassthroughLeanredNoiseBlock

import numpy as np

DEFAULT_CONV_ARGS = dict(kernel_size=3,padding="same", kernel_initializer="he_normal", use_bias=False)
def flatten(l):
    return [item for sublist in l for item in sublist]

class ProgressivePassthroughGenerator(PassthroughLeanredNoiseBlock):
    instance_count = 0
    def __init__(self,
                 latent_noise_size: int,
                 label_size: int,
                 passthrough_adder_initializer =   Constant(0.95),
                 previous_generator: ProgressivePassthroughGenerator = None,
                 name: str = None):
        super(ProgressivePassthroughGenerator, self).__init__(passthrough_adder_initializer, 
                                                              upsample_size=(2,2),
                                                              name= name if name is not None else f"prog_passthrough_generator_{ProgressivePassthroughGenerator.instance_count}")
        ProgressivePassthroughGenerator.instance_count += 1
        self.label_size = label_size
        self.latent_noise_size = latent_noise_size
        self.previous_generator = previous_generator
    
    def call(self, input_tensor, training=True):
        # print("-"*10,self.__class__.__name__,"-"*20)
        # print(f"{self.__class__.__name__} input: ", [y.shape for y in input_tensor])
        prev_generation, pg_extra_outs = self.previous_generator(input_tensor, training=training)
        # print(f"{self.__class__.__name__} pg_out: ",prev_generation.shape, [y.shape if hasattr(y, "shape") else len(y) for y in pg_extra_outs])
        x, extra_outs = super().call(prev_generation, training)
        # print(f"{self.__class__.__name__} out: ", x.shape, [y.shape if hasattr(y, "shape") else len(y) for y in extra_outs])
        return x, [*extra_outs, *pg_extra_outs]
    
    @property
    def input_shape(self):
        return [(self.latent_noise_size,), (self.label_size,)]
    
    def get_config(self):
        return  {
            "latent_noise_size": self.latent_noise_size,
            "label_size": self.label_size,
            "passthrough_adder_initialization": self.passthrough_adder_initializer,
            "previous_generator": self.previous_generator.get_config()
        }

class ProgressiveGenerator224x224(ProgressivePassthroughGenerator):
    instance_count = 0
    def __init__(self, 
                 latent_noise_size: int, 
                 label_size: int,
                 passthrough_adder_initializer = Constant(0.95),
                 previous_generator: ProgressivePassthroughGenerator = None,
                 name: str = None):
        super(ProgressiveGenerator224x224, self).__init__(
            latent_noise_size,
            label_size, 
            passthrough_adder_initializer,
            previous_generator if previous_generator is not None else ProgressiveGenerator112x112(latent_noise_size, label_size, passthrough_adder_initializer),
            name= name if name is not None else f"prog_generator_224x224_{ProgressiveGenerator224x224.instance_count}")
        ProgressiveGenerator224x224.instance_count += 1
        self.layer_definitions = [
            UpSampling2D(name=f"{self.name}_upsampler"),
            Conv2DAutoAdaNormLeakyReluBlock(2, (None, 224, 224, 3), 0.2, dict(filters=64, **DEFAULT_CONV_ARGS), name=self.name),
            Conv2DBatchNormLeakyReluBlock(1, (None, 224, 224, 64), 0.2, dict(filters=3, **DEFAULT_CONV_ARGS), name=self.name)
        ]

class ProgressiveGenerator112x112(ProgressivePassthroughGenerator):
    instance_count = 0
    def __init__(self, 
                 latent_noise_size: int, 
                 label_size: int,
                 passthrough_adder_initializer = Constant(0.95),
                 previous_generator: ProgressivePassthroughGenerator = None,
                 name: str = None):
        super(ProgressiveGenerator112x112, self).__init__(
            latent_noise_size, 
            label_size,
            passthrough_adder_initializer,
            previous_generator if previous_generator is not None else ProgressiveGenerator56x56(latent_noise_size, label_size, passthrough_adder_initializer),
            name= name if name is not None else f"prog_generator_112x112_{ProgressiveGenerator112x112.instance_count}")
        ProgressiveGenerator112x112.instance_count += 1
        self.layer_definitions = [
            UpSampling2D(name=f"{self.name}_upsampler"),
            Conv2DAutoAdaNormLeakyReluBlock(2, (None, 112, 112, 3),  0.2, dict(filters=128, **DEFAULT_CONV_ARGS), name=self.name),
            Conv2DBatchNormLeakyReluBlock(1, (None, 112, 112, 128),  0.2, dict(filters=3, **DEFAULT_CONV_ARGS), name=self.name)
        ]
    
class ProgressiveGenerator56x56(ProgressivePassthroughGenerator):
    instance_count = 0
    def __init__(self, 
                 latent_noise_size: int, 
                 label_size: int, 
                 passthrough_adder_initializer = Constant(0.95),
                 previous_generator: ProgressivePassthroughGenerator = None,
                 name: str = None):
        super(ProgressiveGenerator56x56, self).__init__(
            latent_noise_size, 
            label_size,
            passthrough_adder_initializer,
            previous_generator if previous_generator is not None else ProgressiveGenerator28x28(latent_noise_size, label_size, passthrough_adder_initializer),
            name= name if name is not None else f"prog_generator_56x56_{ProgressiveGenerator56x56.instance_count}")
        ProgressiveGenerator56x56.instance_count += 1
        self.layer_definitions = [
            UpSampling2D(name=f"{self.name}_upsampler"),
            Conv2DAutoAdaNormLeakyReluBlock(3, (None, 56, 56, 3),  0.2, dict(filters=256, **DEFAULT_CONV_ARGS), name=self.name),
            Conv2DBatchNormLeakyReluBlock(1, (None, 56, 56, 256),   0.2, dict(filters=3, **DEFAULT_CONV_ARGS), name=self.name)
        ]

class ProgressiveGenerator28x28(ProgressivePassthroughGenerator):
    instance_count = 0
    def __init__(self, 
                 latent_noise_size: int, 
                 label_size: int, 
                 passthrough_adder_initializer = Constant(0.95),
                 previous_generator: ProgressivePassthroughGenerator = None,
                 name: str = None):
        super(ProgressiveGenerator28x28, self).__init__(
            latent_noise_size,
            label_size,
            passthrough_adder_initializer,
            previous_generator if previous_generator is not None else ProgressiveGenerator14x14(latent_noise_size, label_size, passthrough_adder_initializer),
            name= name if name is not None else f"prog_generator_28x28_{ProgressiveGenerator28x28.instance_count}")
        ProgressiveGenerator28x28.instance_count += 1
        self.layer_definitions = [
            UpSampling2D(name=f"{self.name}_upsampler"),
            Conv2DAutoAdaNormLeakyReluBlock(3, (None, 28, 28, 3),   0.2, dict(filters=512, **DEFAULT_CONV_ARGS), name=self.name),
            Conv2DBatchNormLeakyReluBlock(1, (None, 28, 28, 512),  0.2, dict(filters=3, **DEFAULT_CONV_ARGS), name=self.name)
        ]

class ProgressiveGenerator14x14(ProgressivePassthroughGenerator):
    instance_count = 0
    def __init__(self, 
                 latent_noise_size: int,
                 label_size: int, 
                 passthrough_adder_initializer = Constant(0.95),
                 name: str = None):
        super(ProgressiveGenerator14x14, self).__init__(
            latent_noise_size,
            label_size, 
            passthrough_adder_initializer,
            ProgressiveGenerator7x7(latent_noise_size, label_size),
            name= name if name is not None else f"prog_generator_14x14_{ProgressiveGenerator14x14.instance_count}") 
        ProgressiveGenerator14x14.instance_count += 1
        self.layer_definitions = [
            UpSampling2D(name=f"{self.name}_upsampler"),
            Conv2DAutoAdaNormLeakyReluBlock(3, (None, 14, 14, 3),  0.2, dict(filters=512, **DEFAULT_CONV_ARGS), name=self.name),
            Conv2DBatchNormLeakyReluBlock(1, (None, 14, 14, 512),  0.2, dict(filters=3, **DEFAULT_CONV_ARGS), name=self.name)
        ]

class ProgressiveGenerator7x7(GraphableModelBlock):
    instance_count = 0
    def __init__(self, latent_noise_size: int, label_size: int, name: str = None):
        super(ProgressiveGenerator7x7, self).__init__(name= name if name is not None else f"prog_generator_7x7_{ProgressiveGenerator7x7.instance_count}")
        ProgressiveGenerator7x7.instance_count += 1
        self.latent_noise_size = latent_noise_size
        self.label_size = label_size
        self.label_adder = WeightedAdd(initializer=Constant(0.5), name=f"{self.name}_label_adder")
        self.label_dense = Dense(latent_noise_size, activation='sigmoid', name=f"{self.name}_label_dense")
        self.layer_definitions = [
            Dense(latent_noise_size), 
            Dense(7*7*32, activation='sigmoid'), 
            Dense(7*7*32, activation='sigmoid'), 
            Reshape((7,7,32)),
            Conv2DAutoAdaNormLeakyReluBlock(3, (None, 7, 7, 32),  0.2,  dict(filters=512, **DEFAULT_CONV_ARGS), name=self.name),
            Conv2DBatchNormLeakyReluBlock(1, (None, 7, 7, 512),  0.2, dict(filters=3,   **DEFAULT_CONV_ARGS), name=self.name)
        ]
    
    @property
    def input_shape(self):
        return [(self.latent_noise_size,), (self.label_size,)]
    
    def call(self, input_tensors, training=True):
        # print("-"*10,self.__class__.__name__, "-"*20)
        # print(f"{self.__class__.__name__} input: ", [x.shape for x in input_tensors])
        input_tensor, label = input_tensors
        label_tensor = self.label_dense(label, training=training)
        x = self.label_adder([input_tensor, label_tensor], training=training)
        x, extra_outs_1 = super().call(x, training=training)
        # print(f"{self.__class__.__name__} out: ", x.shape, extra_outs_1)
        return x, extra_outs_1
    
    def get_config(self):
        return  {
            "latent_noise_size": self.latent_noise_size,
            "label_size": self.label_size,
            "name": self.name,
        }
