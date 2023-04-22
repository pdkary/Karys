from keras.layers import MaxPooling2D, UpSampling2D
from karys.layers.LearnedNoise import LearnedNoise
from karys.layers.WeightedAdd import WeightedAdd
from karys.models.bases import GraphableModelBlock
from keras.initializers import Constant

class PassthroughBlock(GraphableModelBlock):
    instance_count = 0
    def __init__(self, passthrough_adder_initializer = Constant(0.95), pool_size = (1,1), upsample_size = (1,1), name: str = None):
        super(PassthroughBlock, self). __init__(name = name if name is not None else f"passthrough_block_{PassthroughBlock.instance_count}")
        self.pool_size = pool_size
        self.upsample_size = upsample_size
        self.passthrough_adder_initializer = passthrough_adder_initializer
        self.passthrough_adder = WeightedAdd(initializer=passthrough_adder_initializer, name=f"{self.name}_passthrough_adder")
        self.upsampler = None if upsample_size == (1,1) else UpSampling2D(size=upsample_size, name=f"{self.name}_pooler")
        self.pooler = None if pool_size == (1,1) else MaxPooling2D(pool_size=pool_size, name=f"{self.name}_pooler")

    def call(self, input_tensor, training=True):
        passthrough_tensor = input_tensor
        if self.upsampler is not None:
            passthrough_tensor = self.upsampler(passthrough_tensor, training=training)
        if self.pooler is not None:
            passthrough_tensor = self.pooler(passthrough_tensor, training=training)

        x, extra_outs = super().call(input_tensor, training)
        x = self.passthrough_adder([x, passthrough_tensor], training=training)
        return x, [input_tensor, *extra_outs]
    
    def get_config(self):
        return  {
            "pool_size": self.pool_size,
            "upsample_size": self.upsample_size,
            "passthrough_adder_initialization": self.passthrough_adder_initializer
        }

class PassthroughLeanredNoiseBlock(PassthroughBlock):
    instance_count = 0
    def __init__(self, passthrough_adder_initializer = Constant(0.95), pool_size = (1,1), upsample_size = (1,1), name: str = None):
        super(PassthroughLeanredNoiseBlock, self). __init__(passthrough_adder_initializer, pool_size, upsample_size, 
                                                            name=name if name is not None else f"passthrough_learned_noise_block_{PassthroughLeanredNoiseBlock.instance_count}")
        self.noise_adder = LearnedNoise(name=f"{self.name}_learned_noise")

    def call(self, input_tensor, training=True):
        x, extra_outs = super().call(input_tensor, training)
        x = self.noise_adder(x, training=training)
        return x, extra_outs