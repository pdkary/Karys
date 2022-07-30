from typing import Tuple, List
from models.ClassificationModel import ClassificationModel
from models.EncoderModel import EncoderModel
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.losses import Loss
from tensorflow.keras.layers import Layer


class EncodedClassificationModel():
    @classmethod
    def create(cls, 
                input_shape: Tuple,
                category_labels: List[str], 
                encoder_dimensions: int, 
                classifier_layers: List[Layer],
                encoder_layers: List[Layer],
                encoder_optimizer: Optimizer, 
                encoder_loss: Loss,
                classifier_optimizer: Optimizer, 
                classifier_loss: Loss):
        encoder = EncoderModel(input_shape, encoder_dimensions, encoder_layers, encoder_optimizer, encoder_loss)
        classifier = ClassificationModel([encoder_dimensions], category_labels, classifier_layers, classifier_optimizer, classifier_loss)
        return cls(encoder, classifier)

    def __init__(self, encoder: EncoderModel, classifier: ClassificationModel):
        self.encoder = encoder
        self.classifier = classifier
    
    def encode(self, input_batch, training=False):
        return self.encoder.encode(input_batch, training=training)

    def classify(self, input_batch, training=False):
        _, encoded_batch = self.encode(input_batch, training=training)
        return self.classifier.classify(encoded_batch, training=training)
    
    def build(self, name: str = None, silent=False):
        c_name = None if name is None else name + "_classifier"
        e_name = None if name is None else name + "_encoder"
        self.classifier.build(c_name, silent=silent)
        self.encoder.build(e_name, silent=silent)
