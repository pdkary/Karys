from keras.layers import Layer
import keras.backend as K

class AdaptiveInstanceNormalization(Layer):
    def __init__(self, alpha=1.0):
        super(AdaptiveInstanceNormalization, self).__init__()
        self.alpha = alpha
        self.axes = [1,2]
    
    def call(self, inputs):
        content_input, style_input = inputs
        content_mean, content_std = K.mean(content_input, self.axes, keepdims=True), K.std(content_input, self.axes, keepdims=True)
        style_mean, style_std = K.mean(style_input, self.axes, keepdims=True), K.std(style_input, self.axes, keepdims=True)
        normalized_content = style_std*(content_input - content_mean)/(content_std + 1e-5) + style_mean
        normalized_content = self.alpha*normalized_content + (1-self.alpha)*content_input
        return normalized_content

class AutoAdaptiveInstanceNormalization(Layer):
    counter = 0
    def __init__(self, name="auto_ada_norm", trainable=True, dtype='float32'):
        super(AutoAdaptiveInstanceNormalization, self).__init__(name=f"{name}_{str(AutoAdaptiveInstanceNormalization.counter)}")
        self.axes = [1,2]
        self.trainable = trainable
        self.datatype = dtype
        AutoAdaptiveInstanceNormalization.counter += 1
    
    def build(self, input_shape) -> None:
        i_shape = (1, 1, input_shape[-1])
        self.beta = self.add_weight(
            name="beta",
            shape=i_shape,
            initializer="zeros",
            dtype=self.datatype,
            trainable=self.trainable
        )

        self.gamma = self.add_weight(
            name="gamma",
            shape=i_shape,
            initializer="ones",
            dtype=self.datatype,
            trainable=self.trainable
        )
    
    def call(self, content_input):
        content_input
        content_mean, content_std = K.mean(content_input, self.axes, keepdims=True), K.std(content_input, self.axes, keepdims=True)
        normalized_content = self.gamma*(content_input - content_mean)/(content_std + 1e-5) + self.beta
        return normalized_content