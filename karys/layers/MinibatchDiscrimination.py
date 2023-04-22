
from typing import Tuple
from keras.layers import Layer, Concatenate
from keras.constraints import Constraint, MinMaxNorm
import tensorflow as tf
import keras.backend as K

class MinibatchDiscrimination(Layer):
    """
    doin my best here...
    https://arxiv.org/pdf/1606.03498v1.pdf
    """
    count = 0
    def __init__(self, num_kernels: int, kernel_dim: int, initializer="he_normal", name: str=None) -> None:
        super(MinibatchDiscrimination, self).__init__(name=name + str(self.count))
        self.count += 1
        self.initializer = initializer
        self.num_kernels = num_kernels
        self.kernel_dim = kernel_dim
    
    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[1], self.num_kernels*self.kernel_dim),
            name=f"{self.name}_kernel",
            initializer=self.initializer,
            trainable=True,
        )
    
    def call(self, inputs):
        M = tf.matmul(inputs,self.kernel)
        M = tf.reshape(M,(-1, self.num_kernels, self.kernel_dim))
        Mi = tf.expand_dims(M, 3)
        Mj = tf.transpose(M,perm=[1,2,0])
        Mj = tf.expand_dims(Mj, 0)

        diff = Mi - Mj
        l1 = tf.reduce_sum(tf.abs(diff), axis=2)
        features = tf.reduce_sum(K.exp(-l1), axis=2)
        return tf.concat([inputs, features], axis=1)
       
    def get_config(self):
        return {"num_kernels": self.num_kernels, "kernel_dim": self.kernel_dim, "initializer": self.initializer, "name": self.name}