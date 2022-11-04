from keras.layers import Layer
import tensorflow as tf

class WeightedAdd(Layer):
    def __init__(self) -> None:
        super(WeightedAdd, self).__init__()
    
    def build(self, input_shape) -> None:
        self.a = self.add_weight(
            name=str(input_shape) + "weight",
            initializer='ones',
            dtype='float32',
            trainable=True,
        )
    
    def call(self, inputs):
        assert len(inputs) == 2
        return self.a*inputs[0] + (1 - self.a)*inputs[1]
