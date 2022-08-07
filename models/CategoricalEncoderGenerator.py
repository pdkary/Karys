

from models.EncoderModel import EncoderModel
from models.GenerationModel import GenerationModel
from models.ClassificationModel import ClassificationModel


class CategoricalEncoderGenerator():
    def __init__(self, 
                 encoder: EncoderModel,
                 label_vectorizer: EncoderModel, 
                 decoder: GenerationModel, 
                 classifier: ClassificationModel):
        print("encoder.output_shape: ",encoder.output_shape)
        print("label_vectorizer.output_shape: ",label_vectorizer.output_shape)
        print("decoder.input_shape: ", decoder.input_shape)
        print("classifier.input_shape: ",classifier.input_shape)
        self.encoder: EncoderModel = encoder
        self.label_vectorizer: EncoderModel = label_vectorizer
        self.decoder: GenerationModel = decoder
        self.classifier: ClassificationModel = classifier
    
    def vectorize_labels(self, input_labels, training=False):
        return self.label_vectorizer.encode(input_labels, training=training)
    
    def encode(self, input_batch, training=False):
        return self.encoder.encode(input_batch, training=training)
    
    def classify(self, input_batch, training=False):
        return self.classifier.classify(input_batch, training=training)
    
    def decode(self, input_batch, training=False):
        return self.decoder.generate(input_batch, training=training)
    
    def run_all(self, input_batch, input_labels, training=False):
        #(B, V)
        encoded_vectors = self.encode(input_batch, training=training)
        #(B, V)
        label_image_vectors = self.vectorize_labels(input_labels, training=training)

        #(B, L)
        encoded_image_label_probs, encoded_image_labels = self.classify(encoded_vectors, training=training)
        #(B, L)
        devectorized_label_probs, devectorized_labels = self.classify(label_image_vectors, training=training)

        #(B, W, H, C)
        decoded_images = self.decode(encoded_vectors, training=training)
        #(B, W, H, C)
        decoded_label_images = self.decode(label_image_vectors, training=training)
        
        #(B, V)
        reencoded_image_vectors = self.encode(decoded_images, training=training)
        #(B, V)
        reencoded_label_vectors = self.encode(decoded_label_images, training=training)
        #(B, L)
        reencoded_image_label_probs, reencoded_image_labels  = self.classify(reencoded_image_vectors, training=training)
        #(B, L)
        reencoded_label_image_probs, reencoded_label_image_labels= self.classify(reencoded_label_vectors, training=training)

        return (encoded_vectors, label_image_vectors, reencoded_image_vectors, reencoded_label_vectors,
                encoded_image_label_probs, devectorized_label_probs, reencoded_image_label_probs, reencoded_label_image_probs,
                encoded_image_labels, devectorized_labels, reencoded_image_labels, reencoded_label_image_labels,
                decoded_images, decoded_label_images)
        
    def build(self, name: str = 'CategoricalEncoderGenerator', silent=False):
        e_name = None if name is None else name + "_encoder"
        v_name = None if name is None else name + "_label_vectorizer"
        d_name = None if name is None else name + "_generator"
        c_name = None if name is None else name + "_classifier"
        self.encoder.build(e_name, silent=silent)
        self.label_vectorizer.build(v_name, silent=silent)
        self.decoder.build(d_name, silent=silent)
        self.classifier.build(c_name, silent=silent)