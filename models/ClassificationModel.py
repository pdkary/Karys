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
        category_dim = len(category_labels)
        super(ClassificationModel, self).__init__(input_shape, [category_dim], layers, optimizer, loss, flatten_input=False)
        self.label_dict = {i:c for i,c in enumerate(category_labels)}
        self.label_to_id = {c:i for i,c in enumerate(category_labels)}
        self.label_generator: CategoricalLabel = CategoricalLabel(label_dim=category_dim, label_dict=self.label_dict)
    
    def classify(self, input_batch, training=False):
        classification_pd = self.model(input_batch, training=training)
        argmax_class = np.argmax(classification_pd, axis=1)
        labels = [self.label_dict[x] for x in argmax_class]
        return input_batch, classification_pd, labels
        
        