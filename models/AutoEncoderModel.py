from models.EncoderModel import EncoderModel
from models.ClassificationModel import ClassificationModel

from models.GenerationModel import GenerationModel


class AutoEncoderModel():
    def __init__(self, encoder: EncoderModel, decoder: GenerationModel, classifier: ClassificationModel):
        print("encoder.output_shape: ",encoder.output_shape)
        print("decoder.input_shape: ", decoder.input_shape)
        print("classifier.input_shape: ",classifier.input_shape)
        assert encoder.output_shape == [decoder.input_shape] and encoder.output_shape == classifier.input_shape
        self.encoder: EncoderModel = encoder
        self.decoder: GenerationModel = decoder
        self.classifier: ClassificationModel = classifier

    def encode(self, input_batch, training=False):
        return self.encoder.encode(input_batch, training=training)
    
    def classify(self, input_batch, training=False):
        return self.classifier.classify(input_batch, training=training)
    
    def decode(self, input_batch, training=False):
        return self.decoder.generate(input_batch, training=training)

    def run_all(self, input_batch, training=False):
        encoded_vectors = self.encode(input_batch, training=training)
        encoded_probs, encoded_labels = self.classify(encoded_vectors, training=training)
        decoded_batch = self.decode(encoded_vectors, training=training)
        reencoded_vectors = self.encode(decoded_batch, training=training)
        reencoded_probs, reencoded_labels = self.classify(reencoded_vectors, training=training)
        return encoded_vectors, encoded_probs, encoded_labels, decoded_batch, reencoded_vectors, reencoded_probs, reencoded_labels

    def build(self, name: str = None, silent=False):
        e_name = None if name is None else name + "_encoder"
        d_name = None if name is None else name + "_generator"
        c_name = None if name is None else name + "_classifier"
        self.encoder.build(e_name, silent=silent)
        self.decoder.build(d_name, silent=silent)
        self.classifier.build(c_name, silent=silent)
