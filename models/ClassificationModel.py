from typing import List, Tuple
import os
import numpy as np
from tensorflow.keras.layers import Layer
from tensorflow.keras.losses import Loss
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Optimizer
from data.labels.CategoricalLabel import CategoricalLabel

from models.ModelWrapper import ModelWrapper

class ClassificationModel(ModelWrapper):
    def __init__(self, 
                 input_shape: Tuple,
                 category_labels: List[str], 
                 layers: List[Layer], 
                 optimizer: Optimizer, 
                 loss: Loss,
                 model: Model = None):
        super(ClassificationModel, self).__init__(input_shape, [len(category_labels)], layers, optimizer, loss, model=model)
        self.label_generator: CategoricalLabel = CategoricalLabel(category_labels)
    
    @classmethod
    def load_from_filepath(cls, filepath, category_labels: List[str], optimizer: Optimizer, loss: Loss):
        filepath = os.path.normpath(filepath)
        model: Model = load_model(filepath)
        return cls(model.input_shape, category_labels, model.layers, optimizer, loss, model)
    
    def classify(self, input_batch, training=False):
        classification_pd = self.model(input_batch, training=training)
        argmax_class = np.argmax(classification_pd, axis=1)
        labels = self.label_generator.get_label_names_by_ids(argmax_class)
        return input_batch, classification_pd, labels
        
        