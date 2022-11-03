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

        