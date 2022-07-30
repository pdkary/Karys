from models.EncodedClassificationModel import EncodedClassificationModel

from models.GenerationModel import GenerationModel


class AutoEncoderModel():
    def __init__(self, encoder: EncodedClassificationModel, generator: GenerationModel):
        self.encoder: EncodedClassificationModel = encoder
        self.generator: GenerationModel = generator

    def encode(self, input_batch, training=False):
        return self.encoder.encode(input_batch, training=training)
    
    def classify(self, input_batch, training=False):
        _, encoded_batch = self.encode(input_batch, training=training)
        return self.encoder.classify(encoded_batch, training=training)
    
    def generate(self, noise_batch, training=False):
        return self.generator.generate(noise_batch, training=training)

    def build(self, name: str = None, silent=False):
        c_name = None if name is None else name + "_classifier"
        g_name = None if name is None else name + "_generator"
        self.encoder.build(c_name, silent=silent)
        self.generator.build(g_name, silent=silent)
