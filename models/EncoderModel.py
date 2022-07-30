from typing import List, Tuple
import numpy as np
from tensorflow.keras.layers import Layer
from tensorflow.keras.losses import Loss
from tensorflow.keras.optimizers import Optimizer

from models.ModelWrapper import ModelWrapper

class EncoderModel(ModelWrapper):
    def __init__(self, 
                 input_shape: Tuple,
                 vector_size: int, 
                 layers: List[Layer], 
                 optimizer: Optimizer, 
                 loss: Loss):
        super(EncoderModel, self).__init__(input_shape, [vector_size], layers, optimizer, loss, flatten_input=False)
    
    def encode(self, input_batch, training=False):
        encoded_batch = self.model(input_batch, training=training)
        return input_batch, encoded_batch
        
        