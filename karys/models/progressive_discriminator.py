from models.bases import GraphableModelBlock
from models.blocking import Conv2DBatchNormLeakyReluBlock
from keras.layers import MaxPooling2D, Flatten, Dense, Activation, Concatenate

class ProgressiveDiscriminator(GraphableModelBlock):
    def __init__(self, feature_size: int = 4096, downsample_input = False):
        super(ProgressiveDiscriminator, self).__init__()
        self.downsample_input = downsample_input
        self.prog_gan_224 = ProgressiveDiscriminator224x224(feature_size)
        self.prog_gan_112 = ProgressiveDiscriminator112x112(feature_size)
        self.prog_gan_56 = ProgressiveDiscriminator56x56(feature_size)
        self.prog_gan_28 = ProgressiveDiscriminator28x28(feature_size)
        self.prog_gan_14 = ProgressiveDiscriminator14x14(feature_size)
        self.prog_gan_7 = ProgressiveDiscriminator7x7(feature_size)

    @property
    def input_shape(self):
        if self.downsample_input:
            return [(224,224,3)]
        else:
            return [(224,224,3),(112,112,3),(56,56,3),(28,28,3),(14,14,3),(7,7,3)]
    
    def call(self, input_tensor, training=False):
        if self.downsample_input:
            out_224 = self.prog_gan_224(input_tensor, training)
            smaller_tensor = MaxPooling2D()(input_tensor)
            out_112 = self.prog_gan_112(smaller_tensor, training)
            smaller_tensor = MaxPooling2D()(smaller_tensor)
            out_56 = self.prog_gan_56(smaller_tensor, training)
            smaller_tensor = MaxPooling2D()(smaller_tensor)
            out_28 = self.prog_gan_28(smaller_tensor, training)
            smaller_tensor = MaxPooling2D()(smaller_tensor)
            out_14 = self.prog_gan_14(smaller_tensor, training)
            smaller_tensor = MaxPooling2D()(smaller_tensor)
            out_7 = self.prog_gan_7(smaller_tensor, training)
        else:
            out_224 = self.prog_gan_224(input_tensor[0], training)
            out_112 = self.prog_gan_112(input_tensor[1], training)
            out_56 = self.prog_gan_56(input_tensor[2], training)
            out_28 = self.prog_gan_28(input_tensor[3], training)
            out_14 = self.prog_gan_14(input_tensor[4], training)
            out_7 = self.prog_gan_7(input_tensor[5], training)
        return Concatenate(axis=0)([out_224, out_112, out_56, out_28, out_14, out_7])

class ProgressiveDiscriminator224x224(GraphableModelBlock):
    def __init__(self, feature_size: int = 4096):
        super(ProgressiveDiscriminator224x224, self).__init__()
        self.layer_definitions = [
            Conv2DBatchNormLeakyReluBlock(2, 0.08, dict(filters=64,kernel_size=3, padding="same")),
            MaxPooling2D(),
            ProgressiveDiscriminator112x112(feature_size)]

    @property
    def input_shape(self):
        return (224,224,3)

class ProgressiveDiscriminator112x112(GraphableModelBlock):
    def __init__(self, feature_size: int = 4096):
        super(ProgressiveDiscriminator112x112, self).__init__()
        self.layer_definitions = [
            Conv2DBatchNormLeakyReluBlock(2, 0.08, dict(filters=128,kernel_size=3, padding="same")),
            MaxPooling2D(),
            ProgressiveDiscriminator56x56(feature_size)]

    @property
    def input_shape(self):
        return (112,112,3)

class ProgressiveDiscriminator56x56(GraphableModelBlock):
    def __init__(self, feature_size: int = 4096):
        super(ProgressiveDiscriminator56x56, self).__init__()
        self.layer_definitions = [
            Conv2DBatchNormLeakyReluBlock(3, 0.08, dict(filters=256,kernel_size=3, padding="same")),
            MaxPooling2D(),
            ProgressiveDiscriminator28x28(feature_size)]

    @property
    def input_shape(self):
        return (56,56,3)

class ProgressiveDiscriminator28x28(GraphableModelBlock):
    def __init__(self, feature_size: int = 4096):
        super(ProgressiveDiscriminator28x28, self).__init__()
        self.layer_definitions = [
            Conv2DBatchNormLeakyReluBlock(3, 0.08, dict(filters=512,kernel_size=3, padding="same")),
            MaxPooling2D(),
            ProgressiveDiscriminator14x14(feature_size)]

    @property
    def input_shape(self):
        return (28,28,3)

class ProgressiveDiscriminator14x14(GraphableModelBlock):
    def __init__(self, feature_size: int = 4096):
        super(ProgressiveDiscriminator14x14, self).__init__()
        self.layer_definitions = [
            Conv2DBatchNormLeakyReluBlock(3, 0.08, dict(filters=512,kernel_size=3, padding="same")),
            MaxPooling2D(),
            ProgressiveDiscriminator7x7(feature_size)]

    @property
    def input_shape(self):
        return (14,14,3)

class ProgressiveDiscriminator7x7(GraphableModelBlock):
    def __init__(self, feature_size: int = 4096):
        super(ProgressiveDiscriminator7x7, self).__init__()
        self.layer_definitions = [
            Flatten(),
            Dense(feature_size), Activation('relu'),
            Dense(feature_size), Activation('relu')]
    
    @property
    def input_shape(self):
        return (7,7,3)
