from karys.models.bases import GraphableModelBlock
from karys.models.blocking import Conv2DBatchNormLeakyReluBlock
from keras.layers import UpSampling2D, Dense, Activation, Concatenate, Reshape

class ProgressiveGenerator(GraphableModelBlock):
    def __init__(self, noise_size: int = 512):
        super(ProgressiveGenerator, self).__init__()
        self.noise_size = noise_size
        self.prog_gan_224 = ProgressiveGenerator224x224(noise_size)
        self.prog_gan_112 = ProgressiveGenerator112x112(noise_size)
        self.prog_gan_56 = ProgressiveGenerator56x56(noise_size)
        self.prog_gan_28 = ProgressiveGenerator28x28(noise_size)
        self.prog_gan_14 = ProgressiveGenerator14x14(noise_size)
        self.prog_gan_7 = ProgressiveGenerator7x7(noise_size)

    @property
    def input_shape(self):
        return (self.noise_size,)
    
    def call(self, input_tensor, training=False):
        out_224 = self.prog_gan_224(input_tensor, training)
        out_112 = self.prog_gan_112(input_tensor, training)
        out_56 = self.prog_gan_56(input_tensor, training)
        out_28 = self.prog_gan_28(input_tensor, training)
        out_14 = self.prog_gan_14(input_tensor, training)
        out_7 = self.prog_gan_7(input_tensor, training)
        return [out_224, out_112, out_56, out_28, out_14, out_7]

class ProgressiveGenerator224x224(GraphableModelBlock):
    def __init__(self, noise_size: int = 512):
        super(ProgressiveGenerator224x224, self).__init__()
        self.noise_size = noise_size
        self.layer_definitions = [
            ProgressiveGenerator112x112(noise_size),
            UpSampling2D(),
            Conv2DBatchNormLeakyReluBlock(3, 0.08, dict(filters=512,kernel_size=3, padding="same")),
            Conv2DBatchNormLeakyReluBlock(1, 0.08, dict(filters=3,kernel_size=3, padding="same"))
        ]

    @property
    def input_shape(self):
        return (self.noise_size,)

class ProgressiveGenerator112x112(GraphableModelBlock):
    def __init__(self, noise_size: int = 512):
        super(ProgressiveGenerator112x112, self).__init__()
        self.noise_size = noise_size
        self.layer_definitions = [
            ProgressiveGenerator56x56(noise_size),
            UpSampling2D(),
            Conv2DBatchNormLeakyReluBlock(3, 0.08, dict(filters=512,kernel_size=3, padding="same")),
            Conv2DBatchNormLeakyReluBlock(1, 0.08, dict(filters=3,kernel_size=3, padding="same"))
        ]

    @property
    def input_shape(self):
        return (self.noise_size,)

class ProgressiveGenerator56x56(GraphableModelBlock):
    def __init__(self, noise_size: int = 512):
        super(ProgressiveGenerator56x56, self).__init__()
        self.noise_size = noise_size
        self.layer_definitions = [
            ProgressiveGenerator28x28(noise_size),
            UpSampling2D(),
            Conv2DBatchNormLeakyReluBlock(3, 0.08, dict(filters=512,kernel_size=3, padding="same")),
            Conv2DBatchNormLeakyReluBlock(1, 0.08, dict(filters=3,kernel_size=3, padding="same"))
        ]

    @property
    def input_shape(self):
        return (self.noise_size,)

class ProgressiveGenerator28x28(GraphableModelBlock):
    def __init__(self, noise_size: int = 512):
        super(ProgressiveGenerator28x28, self).__init__()
        self.noise_size = noise_size
        self.layer_definitions = [
            ProgressiveGenerator14x14(noise_size),
            UpSampling2D(),
            Conv2DBatchNormLeakyReluBlock(3, 0.08, dict(filters=512,kernel_size=3, padding="same")),
            Conv2DBatchNormLeakyReluBlock(1, 0.08, dict(filters=3,kernel_size=3, padding="same"))
        ]

    @property
    def input_shape(self):
        return (self.noise_size,)

class ProgressiveGenerator14x14(GraphableModelBlock):
    def __init__(self, noise_size: int = 512):
        super(ProgressiveGenerator14x14, self).__init__()
        self.noise_size = noise_size
        self.layer_definitions = [
            ProgressiveGenerator7x7(noise_size),
            UpSampling2D(),
            Conv2DBatchNormLeakyReluBlock(2, 0.08, dict(filters=512,kernel_size=3, padding="same")),
            Conv2DBatchNormLeakyReluBlock(1, 0.08, dict(filters=3,kernel_size=3, padding="same"))
        ]

    @property
    def input_shape(self):
        return (self.noise_size,)

class ProgressiveGenerator7x7(GraphableModelBlock):
    def __init__(self, noise_size: int = 512):
        super(ProgressiveGenerator7x7, self).__init__()
        self.noise_size = noise_size
        self.layer_definitions = [
            Dense(noise_size), Activation('relu'),
            Dense(noise_size), Activation('relu'),
            Dense(7*7*3), Activation('relu'),
            Reshape((7,7,3))]
    
    @property
    def input_shape(self):
        return (self.noise_size,)
