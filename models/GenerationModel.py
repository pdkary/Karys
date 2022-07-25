from typing import List, Tuple
from tensorflow.keras.layers import Layer
from tensorflow.keras.losses import Loss
from tensorflow.keras.optimizers import Optimizer

from models.ModelWrapper import ModelWrapper

class GenerationModel(ModelWrapper):
    def __init__(self, 
                 input_shape: Tuple, 
                 output_shape: Tuple, 
                 layers: List[Layer], 
                 optimizer: Optimizer, 
                 loss: Loss):
        super(GenerationModel, self).__init__(input_shape, output_shape, layers, optimizer, loss)

    def generate(self, noise_batch, training=False):
        return self.model(noise_batch, training=training)
