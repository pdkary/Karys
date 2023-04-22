from __future__ import annotations
from typing import List
from karys.layers.MinibatchDiscrimination import MinibatchDiscrimination
from karys.models.bases import GraphableModelBlock
from karys.models.convolutional_blocks import Conv2DBatchNormLeakyReluBlock, Conv2DAutoAdaNormLeakyReluBlock
from keras.layers import MaxPooling2D, Flatten, Dense, Activation, Layer
from keras.initializers import Constant

from karys.models.passthrough_blocks import PassthroughBlock
DEFAULT_CONV_ARGS = dict(kernel_size=3, padding="same", kernel_initializer="he_normal", use_bias=False)

def flatten(l):
    if type(l) == list:
        return [item for sublist in l for item in sublist]
    return [l]

class ProgressivePassthroughDescriminator(PassthroughBlock):
    instance_count = 0
    def __init__(self, feature_size: int = 128,
                 category_size: int = 18,
                 next_discriminator: ProgressivePassthroughDescriminator = None,
                 passthrough_adder_initializer =   Constant(0.95),
                 name: str = None):
        super(ProgressivePassthroughDescriminator, self).__init__(passthrough_adder_initializer, pool_size=(2,2), 
                                                                  name = name if name is not None else f"prog_passthrough_descriminator_{ProgressivePassthroughDescriminator.instance_count}")
        ProgressivePassthroughDescriminator.instance_count += 1 
        self.feature_size = feature_size
        self.category_size = category_size
        self.next_discriminator = next_discriminator
    
    def call(self, input_tensor, training=True):
        # print("-"*10,self.__class__.__name__, "-"*20,"\n")
        # print("input_shape: ", input_tensor.shape)
        prev_generation, extra_outs_1 = super().call(input_tensor, training)
        # print("self out: ", prev_generation.shape, extra_outs_1)
        out, extra_outs_2 = self.next_discriminator(prev_generation, training=training)
        # print("next disc: ", out.shape, extra_outs_2)
        return out, [*extra_outs_1, *extra_outs_2]
    
    def get_config(self):
        return {
            "feature_size": self.feature_size,
            "category_size": self.category_size,
            "next_discriminator": self.next_discriminator.get_config().copy(),
            "passthrough_adder_initialization": self.passthrough_adder_initializer,
            "name": self.name
        }

class ProgressiveDiscriminator224x224(ProgressivePassthroughDescriminator):
    instance_count = 0
    def __init__(self, 
                 feature_size = 128, 
                 category_size = 18, 
                 passthrough_adder_initializer =  Constant(0.95),
                 next_discriminator: ProgressivePassthroughDescriminator = None,
                 name: str = None):
        super(ProgressiveDiscriminator224x224, self).__init__(
            feature_size, 
            category_size, 
            next_discriminator if next_discriminator is not None else ProgressiveDiscriminator112x112(feature_size, category_size, passthrough_adder_initializer), 
            passthrough_adder_initializer,
            name if name is not None else f"prog_discriminator_224x224_{ProgressiveDiscriminator224x224.instance_count}")
        ProgressiveDiscriminator224x224.instance_count += 1
        self.layer_definitions = [
            Conv2DBatchNormLeakyReluBlock(2, (None, 224, 224, 3), 0.2, dict(filters=64,**DEFAULT_CONV_ARGS, name=self.name)),
            Conv2DAutoAdaNormLeakyReluBlock(1, (None, 224, 224, 64),0.2, dict(filters=3,**DEFAULT_CONV_ARGS), name=self.name),
            MaxPooling2D(name=f"{self.name}_max_pooling_2d")]

    @property
    def input_shape(self):
        return (None, 224, 224, 3)

class ProgressiveDiscriminator112x112(ProgressivePassthroughDescriminator):
    instance_count = 0
    def __init__(self, 
                 feature_size = 128, 
                 category_size = 18, 
                 passthrough_adder_initializer =  Constant(0.95),
                 next_discriminator: ProgressivePassthroughDescriminator = None,
                 name: str = None):
        super(ProgressiveDiscriminator112x112, self).__init__(
            feature_size, 
            category_size, 
            next_discriminator if next_discriminator is not None else ProgressiveDiscriminator56x56(feature_size, category_size, passthrough_adder_initializer), 
            passthrough_adder_initializer,
            name if name is not None else f"prog_discriminator_112x112_{ProgressiveDiscriminator224x224.instance_count}")
        ProgressiveDiscriminator112x112.instance_count+=1
        self.layer_definitions = [
            Conv2DBatchNormLeakyReluBlock(2, (None, 112, 112, 3), 0.2, dict(filters=128,**DEFAULT_CONV_ARGS), name=self.name),
            Conv2DAutoAdaNormLeakyReluBlock(1, (None, 112, 112, 128), 0.2, dict(filters=3,**DEFAULT_CONV_ARGS), name=self.name),
            MaxPooling2D(name=f"{self.name}_max_pooling_2d")]
        
    @property
    def input_shape(self):
        return (None, 112, 112, 3)

class ProgressiveDiscriminator56x56(ProgressivePassthroughDescriminator):
    instance_count = 0
    def __init__(self, 
                 feature_size = 128, 
                 category_size = 18, 
                 passthrough_adder_initializer =  Constant(0.95),
                 next_discriminator: ProgressivePassthroughDescriminator = None,
                 name: str = None):
        super(ProgressiveDiscriminator56x56, self).__init__(
            feature_size, 
            category_size, 
            next_discriminator if next_discriminator is not None else ProgressiveDiscriminator28x28(feature_size, category_size, passthrough_adder_initializer), 
            passthrough_adder_initializer,
            name = name if name is not None else f"prog_discriminator_56x56_{ProgressiveDiscriminator56x56.instance_count}")
        ProgressiveDiscriminator56x56.instance_count+=1
        self.layer_definitions = [
            Conv2DBatchNormLeakyReluBlock(3, (None, 56, 56, 3), 0.2, dict(filters=256,**DEFAULT_CONV_ARGS), name=self.name),
            Conv2DAutoAdaNormLeakyReluBlock(1, (None, 56, 56, 256), 0.2, dict(filters=3,**DEFAULT_CONV_ARGS), name=self.name),
            MaxPooling2D(name=f"{self.name}_max_pooling_2d")]

    @property
    def input_shape(self):
        return (None, 56, 56, 3)

class ProgressiveDiscriminator28x28(ProgressivePassthroughDescriminator):
    def __init__(self, 
                 feature_size = 128, 
                 category_size = 18, 
                 passthrough_adder_initializer =  Constant(0.95), 
                 next_discriminator: ProgressivePassthroughDescriminator = None,
                 name: str = None):
        super(ProgressiveDiscriminator28x28, self).__init__(
            feature_size, 
            category_size, 
            next_discriminator if next_discriminator is not None else ProgressiveDiscriminator14x14(feature_size, category_size, passthrough_adder_initializer), 
            passthrough_adder_initializer,
            name = name if name is not None else f"prog_discriminator_28x28_{ProgressiveDiscriminator28x28.instance_count}")
        ProgressiveDiscriminator28x28.instance_count+=1
        
        self.layer_definitions = [
            Conv2DBatchNormLeakyReluBlock(3, (None, 28, 28, 3), 0.2, dict(filters=512,**DEFAULT_CONV_ARGS), name=self.name),
            Conv2DAutoAdaNormLeakyReluBlock(1, (None, 28, 28, 512), 0.2, dict(filters=3,**DEFAULT_CONV_ARGS), name=self.name),
            MaxPooling2D(name=f"{self.name}_max_pooling_2d")]

    @property
    def input_shape(self):
        return (None, 28, 28, 3)

class ProgressiveDiscriminator14x14(ProgressivePassthroughDescriminator):
    instance_count = 0
    def __init__(self, 
                 feature_size = 128, 
                 category_size = 18, 
                 passthrough_adder_initializer =  Constant(0.95),
                 name: str = None):
        super(ProgressiveDiscriminator14x14, self).__init__(
            feature_size, 
            category_size, 
            ProgressiveDiscriminator7x7(feature_size, category_size), 
            passthrough_adder_initializer,
            name = name if name is not None else f"prog_discriminator_14x14_{ProgressiveDiscriminator14x14.instance_count}")
        ProgressiveDiscriminator14x14.instance_count+=1
        self.layer_definitions = [
            Conv2DBatchNormLeakyReluBlock(3, (None, 14, 14, 3), 0.2, dict(filters=512,**DEFAULT_CONV_ARGS), name=self.name),
            Conv2DAutoAdaNormLeakyReluBlock(1, (None, 14, 14, 512), 0.2, dict(filters=3,**DEFAULT_CONV_ARGS), name=self.name),
            MaxPooling2D(name=f"{self.name}_max_pooling_2d")]
    
    @property
    def input_shape(self):
        return (None, 14, 14, 3)

class ProgressiveDiscriminator7x7(GraphableModelBlock):
    instance_count = 0
    def __init__(self, feature_size = 128, category_size = 18, name: str = None):
        super(ProgressiveDiscriminator7x7, self).__init__(name= name if name is not None else f"prog_discriminator_7x7_{ProgressiveDiscriminator7x7.instance_count}")
        ProgressiveDiscriminator7x7.instance_count+=1
        self.feature_size = feature_size
        self.category_size = category_size
        self.categorization_layer = Dense(category_size, activation='softmax')
        self.layer_definitions = [
            Flatten(),
            Dense(feature_size, activation='sigmoid'),
            MinibatchDiscrimination(16, category_size, name=f"{self.name}_minibatch_discriminator"),
            Dense(feature_size, activation='sigmoid', name="feature_output"),]
    
    @property
    def input_shape(self):
        return (None, 7, 7, 3)

    def get_config(self):
        return {
            "feature_size": self.feature_size,
            "category_size": self.category_size,
            "name": self.name,
        }
    
    def call(self, input_tensor, training=True):
        # print("-"*10,self.__class__.__name__, "-"*20)
        # print("input_shape: ", input_tensor.shape)
        features, extra_outs = super().call(input_tensor, training)
        # print("self out: ",features.shape, extra_outs)
        categories = self.categorization_layer(features, training=training)
        # print("categories: ",categories.shape)
        return categories, [*extra_outs, input_tensor, features]