from typing import List, Tuple
import numpy as np
from tensorflow.keras.layers import Layer
from tensorflow.keras.losses import Loss
from tensorflow.keras.optimizers import Optimizer
from data.labels.CategoricalLabel import CategoricalLabel

from models.ModelWrapper import ModelWrapper

class ClassificationModel(ModelWrapper):
    def __init__(self, 
                 input_shape: Tuple,
                 category_labels: List[str], 
                 layers: List[Layer], 
                 optimizer: Optimizer, 
                 loss: Loss):
        super(ClassificationModel, self).__init__(input_shape, [len(category_labels)], layers, optimizer, loss, flatten_input=False)
        self.label_generator: CategoricalLabel = CategoricalLabel(category_labels)
    
    def classify(self, input_batch, training=False):
        classification_pd = self.model(input_batch, training=training)
        argmax_class = np.argmax(classification_pd, axis=1)
        labels = self.label_generator.get_label_names_by_ids(argmax_class)
        return input_batch, classification_pd, labels
        
        