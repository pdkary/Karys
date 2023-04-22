from keras.layers import Layer
from keras.initializers import Constant
from karys.layers.WeightedAdd import WeightedAdd
from keras.constraints import MinMaxNorm
import keras.backend as K

class LearnedNoise(Layer):
    instance_count = 0
    def __init__(self, name: str = None) -> None:
        super(LearnedNoise, self).__init__(name=name if name is not None else f"learned_noise_{LearnedNoise.instance_count}")
        LearnedNoise.instance_count += 1
        self.wa = WeightedAdd(initializer = Constant(0.95), name=f"{self.name}_weighted_add")
    
    def build(self, input_shape) -> None:
        self.noise_weight = self.add_weight(
            name=self.name,
            shape=input_shape[1:],
            initializer='he_normal',
            dtype='float32',
            trainable=True,
            constraint=MinMaxNorm(min_value=0.0, max_value=1.0, rate=0.9, axis=0)
        )
    
    def call(self, inputs):
        ##make sure the noise weight has the same mean and std as the input
        noise = K.std(inputs)*(self.noise_weight - K.mean(self.noise_weight))/(K.std(self.noise_weight) + 1e-5) + K.mean(inputs)
        return self.wa([inputs, noise])
    
    def get_config(self):
        return {"name": self.name}

    @property
    def a(self):
        return self.wa.a
