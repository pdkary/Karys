from models.ClassificationModel import ClassificationModel
from models.EncoderModel import EncoderModel


class EncodedClassificationModel():
    def __init__(self, encoder: EncoderModel, classifier: ClassificationModel):
        self.encoder = encoder
        self.classifier = classifier
    
    def encode_and_classify(self, input_batch, training=False):
        encoded_batch = self.encoder.encode(input_batch, training=training)
        return encoded_batch, self.classifier.classify(encoded_batch, training=training)
    
    def build(self, name: str = "EC", silent=False):
        c_name = None if name is None else name + "_classifier"
        e_name = None if name is None else name + "_encoder"
        self.classifier.build(c_name, silent=silent)
        self.encoder.build(e_name, silent=silent)
