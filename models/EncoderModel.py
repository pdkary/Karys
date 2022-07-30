from typing import List, Tuple
import os
import glob
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.losses import Loss
from tensorflow.keras.optimizers import Optimizer

from models.ModelWrapper import ModelWrapper

class EncoderModel(ModelWrapper):
    def __init__(self, 
                 input_shape: Tuple,
                 vector_size: int, 
                 layers: List[Layer], 
                 optimizer: Optimizer, 
                 loss: Loss,
                 model: Model = None):
        super(EncoderModel, self).__init__(input_shape, [vector_size], layers, optimizer, loss, model = model)

    @classmethod
    def load_from_filepath(cls, filepath, optimizer: Optimizer, loss: Loss):
        filepath = os.path.abspath(filepath)
        model: Model = load_model(filepath)
        return cls(model.input_shape, model.output_shape[-1], model.layers, optimizer, loss, model = model)
    
    def encode(self, input_batch, training=False):
        encoded_batch = self.model(input_batch, training=training)
        return input_batch, encoded_batch
        
        