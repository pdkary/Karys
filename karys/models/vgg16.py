from karys.models.bases import GraphableModelBlock
from karys.models.convolutional_blocks import Conv2DBatchNormLeakyReluBlock
from keras.layers import MaxPooling2D, Flatten, Dense, Activation,  Reshape, UpSampling2D, Conv2D

class Vgg16(GraphableModelBlock):
    def __init__(self, feature_size: int = 4096):
        super(Vgg16, self).__init__()
        self.layer_definitions = [
            Conv2DBatchNormLeakyReluBlock(2, (None, 224, 224, 3), 0.08 , dict(filters=64,kernel_size=3, padding="same")),
            MaxPooling2D(),
            Conv2DBatchNormLeakyReluBlock(2, (None, 112, 112, 64), 0.08, dict(filters=128,kernel_size=3, padding="same")),
            MaxPooling2D(),
            Conv2DBatchNormLeakyReluBlock(3, (None, 56, 56, 128), 0.08, dict(filters=256,kernel_size=3, padding="same")),
            MaxPooling2D(),
            Conv2DBatchNormLeakyReluBlock(3, (None, 28, 28, 256), 0.08, dict(filters=512,kernel_size=3, padding="same")),
            MaxPooling2D(),
            Conv2DBatchNormLeakyReluBlock(3, (None, 14, 14, 512), 0.08, dict(filters=512,kernel_size=3, padding="same")),
            MaxPooling2D(),
            Flatten(),
            Dense(feature_size), Activation('relu'),
            Dense(feature_size), Activation('relu'),
        ]
    
    @property
    def input_shape(self):
        return (224,224,3)

    def call(self, input_tensor, training=False):
        x = input_tensor
        for layer in self.layer_definitions:
            x = layer(x, training = training)
        return x

class Vgg16Lite(GraphableModelBlock):
    def __init__(self, feature_size: int = 4096):
        super(Vgg16Lite, self).__init__()
        self.layer_definitions = [
            Conv2DBatchNormLeakyReluBlock(2, (None, 224, 224, 3), 0.08, dict(filters=64,kernel_size=3, padding="same")),
            MaxPooling2D(),
            Conv2DBatchNormLeakyReluBlock(2, (None, 112, 112, 64), 0.08, dict(filters=128,kernel_size=3, padding="same")),
            MaxPooling2D(),
            Conv2DBatchNormLeakyReluBlock(3, (None, 56, 56, 128), 0.08, dict(filters=256,kernel_size=3, padding="same")),
            MaxPooling2D(),
            Conv2DBatchNormLeakyReluBlock(3, (None, 28, 28, 256), 0.08, dict(filters=512,kernel_size=3, padding="same")),
            MaxPooling2D(),
            Conv2DBatchNormLeakyReluBlock(3, (None, 14, 14, 512), 0.08, dict(filters=512,kernel_size=3, padding="same")),
            MaxPooling2D(),
            Conv2DBatchNormLeakyReluBlock(3, (None, 7, 7, 512), 0.08, dict(filters=512,kernel_size=3, padding="same")),
            MaxPooling2D(),
            Flatten(),
            Dense(feature_size), Activation('relu'),
            Dense(feature_size), Activation('relu'),
        ]
    
    @property
    def input_shape(self):
        return (224,224,3)

    def call(self, input_tensor, training=False):
        x = input_tensor
        for layer in self.layer_definitions:
            x = layer(x, training = training)
        return x

class Vgg16Classifier(GraphableModelBlock):
    def __init__(self, num_classifications, output_features=True, final_activation=None):
        super(Vgg16Classifier, self).__init__()
        self.vgg16 = Vgg16()
        self.output_features = output_features
        self.classification_layer = Dense(num_classifications)
        self.classification_activation = Activation('softmax') if final_activation is None else final_activation
    
    @property
    def input_shape(self):
        return (224,224,3)

    def call(self, input_tensor, training=False):
        features = self.vgg16.call(input_tensor, training=training)
        classifications = self.classification_activation(self.classification_layer(features))
        if self.output_features:
            return (features, classifications)
        else:
            return classifications

class Vgg16LiteClassifier(GraphableModelBlock):
    def __init__(self, num_classifications, output_features=True, final_activation=None):
        super(Vgg16LiteClassifier, self).__init__()
        self.vgg16 = Vgg16Lite()
        self.output_features = output_features
        self.classification_layer = Dense(num_classifications)
        self.classification_activation = Activation('softmax') if final_activation is None else final_activation
    
    @property
    def input_shape(self):
        return (224,224,3)

    def call(self, input_tensor, training=False):
        features = self.vgg16.call(input_tensor, training=training)
        classifications = self.classification_activation(self.classification_layer(features))
        if self.output_features:
            return (features, classifications)
        else:
            return classifications

class ReverseVgg16Generator(GraphableModelBlock):
    def __init__(self, noise_size: int = 4096):
        super(ReverseVgg16Generator, self).__init__()
        self.noise_size = noise_size
        self.layer_definitions = [
            Dense(noise_size), Activation('tanh'),
            Dense(7*7*512),Activation('relu'),
            Reshape((7,7,512)),
            UpSampling2D(),
            Conv2DBatchNormLeakyReluBlock(3, (None, 14, 14, 512), 0.08, dict(filters=512,kernel_size=3, padding="same")),
            UpSampling2D(),
            Conv2DBatchNormLeakyReluBlock(3, (None, 28, 28, 512), 0.08, dict(filters=512,kernel_size=3, padding="same")),
            UpSampling2D(),
            Conv2DBatchNormLeakyReluBlock(3, (None, 56, 56, 512), 0.08, dict(filters=256,kernel_size=3, padding="same")),
            UpSampling2D(),
            Conv2DBatchNormLeakyReluBlock(2, (None, 112, 112, 256), 0.08, dict(filters=128,kernel_size=3, padding="same")),
            UpSampling2D(),
            Conv2DBatchNormLeakyReluBlock(2, (None, 224, 224, 128), 0.08, dict(filters=64,kernel_size=3, padding="same")),
            Conv2D(3,3, padding="same"), 
            Activation('sigmoid')
        ]
    
    @property
    def input_shape(self):
        return (self.noise_size,)

    def call(self, input_tensor, training=False):
        x = input_tensor
        for layer in self.layer_definitions:
            x = layer(x, training = training)
        return x