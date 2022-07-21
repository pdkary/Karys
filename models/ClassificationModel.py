from typing import List, Tuple
from tensorflow.keras.layers import Flatten, Input, Layer, Reshape, Dense
from tensorflow.keras.losses import Loss
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer
from data.labels.CategoricalLabel import BatchedCategoricalLabel

from models.ModelWrapper import ModelWrapper

class ClassificationModel(ModelWrapper):
    def __init__(self, 
                 input_shape: Tuple, 
                 category_dim: int, 
                 layers: List[Layer], 
                 optimizer: Optimizer, 
                 loss: Loss):
        super(ClassificationModel, self).__init__(input_shape, [category_dim], layers, optimizer, loss)
        self.label_generator = BatchedCategoricalLabel(label_dim=category_dim)