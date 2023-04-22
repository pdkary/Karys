from keras.layers import Layer
from keras.constraints import Constraint, MinMaxNorm
import tensorflow as tf
import keras.backend as K

class UnitInterval(Constraint):
    """Constrains the weights to be between 0 and 1.
    """
    def __call__(self, w):
        is_le_1 = tf.less_equal(w, 1.0)
        is_ge_0 = tf.greater_equal(w, 0.0)
        is_unit = tf.logical_and(is_ge_0, is_le_1)
        return w*tf.cast(is_unit, K.floatx())
        

class WeightedAdd(Layer):
    """Adds a weighted sum to the inputs.
    """
    count = 0
    def __init__(self, initializer = "ones", name="weighted_add") -> None:
        super(WeightedAdd, self).__init__(name=name + str(self.count))
        self.count += 1
        self.initializer = initializer
        self.a = self.add_weight(
            shape=(1,),
            name=self.name,
            initializer=self.initializer,
            dtype='float32',
            trainable=True,
            constraint=MinMaxNorm(min_value=0.0, max_value=1.0, rate=1.0, axis=0)
        )
    
    
    def call(self, inputs):
        assert len(inputs) == 2
        return self.a*inputs[0] + (1 - self.a)*inputs[1]
    
    def get_config(self):
        return {"initializer": self.initializer, "name": self.name}
