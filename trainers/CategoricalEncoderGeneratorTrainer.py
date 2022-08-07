import numpy as np
import tensorflow as tf

from data.wrappers.ImageDataWrapper import ImageDataWrapper
from models.CategoricalEncoderGenerator import CategoricalEncoderGenerator

def batch(ndarr, batch_size):
    N = len(ndarr)//batch_size
    return np.array(np.split(ndarr[:N*batch_size], N))

def batch_dict(adict, batch_size):
    N = len(adict)//batch_size
    batches = []
    for i in range(N):
        b_names = list(adict.keys())[i*batch_size:(i+1)*batch_size]
        b_vals = np.array(list(adict.values())[i*batch_size:(i+1)*batch_size])
        batches.append((b_names, b_vals))
    return batches

class CategoricalEncoderGeneratorTrainer(object):
    def __init__(self,
                 categorical_encoder_generator: CategoricalEncoderGenerator,
                 labelled_input: ImageDataWrapper,
                 A: float = 0.0, B: float = 0.0,
                 C: float = 0.0, D: float = 0.0):
        self.categorical_encoder_generator = categorical_encoder_generator
        self.labelled_input = labelled_input
        self.A, self.B, self.C, self.D = A, B, C, D

        self.labelled_train_data = self.labelled_input.get_train_dataset()
        self.labelled_test_data = self.labelled_input.get_validation_dataset()

        self.most_recent_generated = None
        self.most_recent_label_generated = None

        self.most_recent_real_encoding = None
        self.most_recent_label_encodings = None
        self.most_recent_gen_encoding = None
        self.most_recent_label_gen_encoding = None
        
        self.most_recent_real_classifications = None
        self.most_recent_label_classifications = None
        self.most_recent_gen_classifications = None
        self.most_recent_label_gen_classifications = None

        
    
    def label_vectorizer_loss(self, encoded_image_labels, devectorized_labels):
        return self.categorical_encoder_generator.label_vectorizer.loss(encoded_image_labels, devectorized_labels)

    ##generator should create images that are correctly identified and DO NOT receive the reencoding flag
    def decoder_loss(self, label_vectors, reencoded_image_probs, reencoded_label_image_probs):
        image_loss = self.categorical_encoder_generator.decoder.loss(label_vectors, reencoded_image_probs)
        label_image_loss = self.categorical_encoder_generator.decoder.loss(label_vectors, reencoded_label_image_probs)
        return image_loss + label_image_loss
    
    ##encoder should create vectors that are correctly identified and DO NOT receive the reencoding flag
    def encoder_loss(self, label_vectors, encoded_label_probs):
        return self.categorical_encoder_generator.encoder.loss(label_vectors, encoded_label_probs)
    
    ##classifier should correctly identify reals without reencoding flag, and reencodings with the reencoding flag
    def classifier_loss(self, label_vectors, encoded_image_probs, reencoded_image_probs, reencoded_label_image_probs):
        batch_size = label_vectors.shape[0]
        re_encoded_flag_vector = self.categorical_encoder_generator.classifier.label_generator.get_label_vector_by_category_name("generated")
        batch_reenc_flag_vectors = np.concatenate([re_encoded_flag_vector]*batch_size,axis=-1).T

        ab_labels = self.A*label_vectors + self.B*batch_reenc_flag_vectors
        cd_labels = self.C*label_vectors + self.D*batch_reenc_flag_vectors
        real_classification_loss = self.categorical_encoder_generator.classifier.loss(label_vectors, encoded_image_probs)
        generator_indication_loss = self.categorical_encoder_generator.classifier.loss(ab_labels, reencoded_image_probs)
        label_generator_indication_loss = self.categorical_encoder_generator.classifier.loss(cd_labels, reencoded_label_image_probs)
        return real_classification_loss + generator_indication_loss + label_generator_indication_loss

    def __run_batch__(self, batch_names, batch_data, training=True):
        batch_labels = [ self.labelled_input.image_labels[n] for n in batch_names]
        labels = self.categorical_encoder_generator.classifier.label_generator.get_label_vectors_by_category_names(batch_labels)

        (V, LV, re_v, re_lv, 
         v_probs, lv_probs, re_v_probs, re_lv_probs, 
         v_lbls, lv_lbls, re_v_lbls, re_lv_lbls,
         gen_batch, label_gen_batch) = self.categorical_encoder_generator.run_all(batch_data, labels, training)

        encoder_loss = self.encoder_loss(labels, v_probs)
        vectorizer_loss = self.label_vectorizer_loss(v_probs, lv_probs)
        decoder_loss = self.decoder_loss(labels, re_v_probs, re_lv_probs)
        classifier_loss = self.classifier_loss(labels, v_probs, re_v_probs, re_lv_probs)

        self.most_recent_generated = gen_batch
        self.most_recent_label_generated = label_gen_batch

        self.most_recent_real_encoding = list(zip(batch_data, batch_labels, V))
        self.most_recent_label_encodings = list(zip(label_gen_batch, batch_labels, LV))
        self.most_recent_gen_encoding = list(zip(gen_batch, batch_labels, re_v))
        self.most_recent_label_gen_encoding = list(zip(label_gen_batch, batch_labels, re_lv))
        
        self.most_recent_real_classifications = list(zip(batch_data, batch_labels, v_lbls))
        self.most_recent_label_classifications = list(zip(label_gen_batch, batch_labels, lv_lbls))
        self.most_recent_gen_classifications = list(zip(gen_batch, batch_labels, re_v_lbls))
        self.most_recent_label_gen_classifications = list(zip(label_gen_batch, batch_labels, re_lv_lbls))

        return encoder_loss, vectorizer_loss, decoder_loss, classifier_loss

    def train(self, batch_size, num_batches):
        e_loss_bucket = []
        v_loss_bucket = []
        d_loss_bucket = []
        c_loss_bucket = []
        batched_input = batch_dict(self.labelled_train_data, batch_size)

        np.random.shuffle(batched_input)
        input_dataset = batched_input[:num_batches]

        for batch_names, batch_data in input_dataset:
            with  tf.GradientTape() as e_tape, tf.GradientTape() as v_tape, tf.GradientTape() as c_tape, tf.GradientTape() as d_tape:
                e_loss, v_loss, d_loss, c_loss = self.__run_batch__(batch_names, batch_data, training=True)
                e_loss_bucket.append(e_loss)
                v_loss_bucket.append(v_loss)
                d_loss_bucket.append(d_loss)
                c_loss_bucket.append(c_loss)

                e_grads = e_tape.gradient(e_loss, self.categorical_encoder_generator.encoder.model.trainable_variables)
                v_grads = v_tape.gradient(v_loss, self.categorical_encoder_generator.label_vectorizer.model.trainable_variables)
                d_grads = d_tape.gradient(d_loss, self.categorical_encoder_generator.decoder.model.trainable_variables)
                c_grads = c_tape.gradient(c_loss, self.categorical_encoder_generator.classifier.model.trainable_variables)
                
                self.categorical_encoder_generator.encoder.optimizer.apply_gradients(zip(e_grads, self.categorical_encoder_generator.encoder.model.trainable_variables))
                self.categorical_encoder_generator.label_vectorizer.optimizer.apply_gradients(zip(v_grads, self.categorical_encoder_generator.label_vectorizer.model.trainable_variables))
                self.categorical_encoder_generator.decoder.optimizer.apply_gradients(zip(d_grads, self.categorical_encoder_generator.decoder.model.trainable_variables))
                self.categorical_encoder_generator.classifier.optimizer.apply_gradients(zip(c_grads, self.categorical_encoder_generator.classifier.model.trainable_variables))
        return np.sum(e_loss_bucket, axis=0), np.sum(v_loss_bucket, axis=0), np.sum(d_loss_bucket, axis=0), np.sum(c_loss_bucket, axis=0)
    
    def test(self, batch_size, num_batches):
        e_loss_bucket = []
        v_loss_bucket = []
        d_loss_bucket = []
        c_loss_bucket = []
        batched_input = batch_dict(self.labelled_test_data, batch_size)

        np.random.shuffle(batched_input)
        input_dataset = batched_input[:num_batches]

        for batch_names, batch_data in input_dataset:
            e_loss, v_loss, d_loss, c_loss = self.__run_batch__(batch_names, batch_data, training=False)
            e_loss_bucket.append(e_loss)
            v_loss_bucket.append(v_loss)
            d_loss_bucket.append(d_loss)
            c_loss_bucket.append(c_loss)
        return np.sum(e_loss_bucket, axis=0), np.sum(v_loss_bucket, axis=0), np.sum(d_loss_bucket, axis=0), np.sum(c_loss_bucket, axis=0)
        
