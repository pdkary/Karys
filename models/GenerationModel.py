import os
from typing import List, Tuple
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.losses import Loss
from tensorflow.keras.optimizers import Optimizer

from models.ModelWrapper import ModelWrapper

class GenerationModel(ModelWrapper):
    def __init__(self, 
                 input_shape: Tuple, 
                 output_shape: Tuple, 
                 layers: List[Layer], 
                 optimizer: Optimizer, 
                 loss: Loss,
                 model: Model = None):
        super(GenerationModel, self).__init__(input_shape, output_shape, layers, optimizer, loss, model = model)
    
    @classmethod
    def load_from_filepath(cls, filepath, optimizer: Optimizer, loss: Loss):
        filepath = os.path.abspath(filepath)
        model: Model = load_model(filepath)
        return cls(model.input_shape, model.output_shape, model.layers, optimizer, loss, model = model)

    def generate(self, noise_batch, training=False):
        return self.model(noise_batch, training=training)
