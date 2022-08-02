import numpy as np
import tensorflow as tf

from data.wrappers.ImageDataWrapper import ImageDataWrapper
from models.AutoEncoderModel import AutoEncoderModel

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

class AutoEncoderTrainer(object):
    def __init__(self,
                 auto_encoder: AutoEncoderModel,
                 labelled_input: ImageDataWrapper):
        self.auto_encoder = auto_encoder
        self.labelled_input = labelled_input

        self.labelled_train_data = self.labelled_input.get_train_dataset()
        self.labelled_test_data = self.labelled_input.get_validation_dataset()

        self.most_recent_generated = None
        self.most_recent_real_encoding = None
        self.most_recent_gen_encoding = None
        self.most_recent_real_classification = None
        self.most_recent_gen_classification = None

    ##generator should create images that are correctly identified and DO NOT receive the reencoding flag
    def decoder_loss(self, label_vectors, reencoded_label_probs):
        return self.auto_encoder.decoder.loss(label_vectors, reencoded_label_probs)
    
    ##encoder should create vectors that are correctly identified and DO NOT receive the reencoding flag
    def encoder_loss(self,label_vectors, encoded_label_probs):
        return self.auto_encoder.encoder.loss(label_vectors, encoded_label_probs)
    
    ##classifier should correctly identify reals without reencoding flag, and reencodings with the reencoding flag
    def classifier_loss(self, label_vectors, encoded_label_probs, reencoded_label_probs):
        batch_size = label_vectors.shape[0]
        re_encoded_flag_vector = self.auto_encoder.classifier.label_generator.get_label_vector_by_category_name("generated")
        batch_reenc_flag_vectors = np.concatenate([re_encoded_flag_vector]*batch_size,axis=-1).T

        flagged_labels = label_vectors + batch_reenc_flag_vectors
        real_classification_loss = self.auto_encoder.classifier.loss(label_vectors, encoded_label_probs)
        generator_indication_loss = self.auto_encoder.classifier.loss(flagged_labels, reencoded_label_probs)
        return real_classification_loss + generator_indication_loss

    def __run_batch__(self, batch_names, batch_data, training=True):
        batch_labels = [ self.labelled_input.image_labels[n] for n in batch_names]
        labels = self.auto_encoder.classifier.label_generator.get_label_vectors_by_category_names(batch_labels)

        V, V_probs, V_lbls, gen_batch, GV, GV_probs, GV_lbls = self.auto_encoder.run_all(batch_data, training)

        decoder_loss = self.decoder_loss(labels, GV_probs)
        encoder_loss = self.encoder_loss(labels, V_probs)
        classifier_loss = self.classifier_loss(labels, V_probs, GV_probs)

        self.most_recent_generated = gen_batch
        self.most_recent_real_encoding = list(zip(batch_data, batch_labels, V))
        self.most_recent_gen_encoding = list(zip(gen_batch, batch_labels, GV))
        self.most_recent_real_classification = list(zip(batch_data, batch_labels, V_lbls))
        self.most_recent_gen_classification = list(zip(gen_batch, batch_labels, GV_lbls))


        # void_label = self.auto_encoder.classifier.label_generator.get_label_vector_by_name("Invalid")
        # void_labels = np.repeat(void_label, batch_data.shape[0],axis=-1).T

        # generator_encoding_loss = self.auto_encoder.decoder.loss(V, GV)
        # generator_classification_loss = self.auto_encoder.decoder.loss(1-void_labels, GV_probs)

        # classifier_real_loss = self.auto_encoder.classifier.loss(labels, V_probs)
        # classifier_gen_loss = self.auto_encoder.classifier.loss(void_labels, GV_probs)

        # generator_loss = generator_encoding_loss + generator_classification_loss
        # classifier_loss = classifier_real_loss + classifier_gen_loss
        # encoder_loss = self.auto_encoder.encoder.loss(void_labels, GV_probs) + self.auto_encoder.encoder.loss(labels, V_probs)
        return encoder_loss, decoder_loss, classifier_loss

    def train(self, batch_size, num_batches):
        e_loss_bucket = []
        d_loss_bucket = []
        c_loss_bucket = []
        batched_input = batch_dict(self.labelled_train_data, batch_size)

        np.random.shuffle(batched_input)
        input_dataset = batched_input[:num_batches]

        for batch_names, batch_data in input_dataset:
            with  tf.GradientTape() as e_tape, tf.GradientTape() as c_tape, tf.GradientTape() as d_tape:
                e_loss, d_loss, c_loss = self.__run_batch__(batch_names, batch_data, training=True)
                e_loss_bucket.append(e_loss)
                d_loss_bucket.append(d_loss)
                c_loss_bucket.append(c_loss)

                e_grads = e_tape.gradient(e_loss, self.auto_encoder.encoder.model.trainable_variables)
                d_grads = d_tape.gradient(d_loss, self.auto_encoder.decoder.model.trainable_variables)
                c_grads = c_tape.gradient(c_loss, self.auto_encoder.classifier.model.trainable_variables)
                
                self.auto_encoder.encoder.optimizer.apply_gradients(zip(e_grads, self.auto_encoder.encoder.model.trainable_variables))
                self.auto_encoder.decoder.optimizer.apply_gradients(zip(d_grads, self.auto_encoder.decoder.model.trainable_variables))
                self.auto_encoder.classifier.optimizer.apply_gradients(zip(c_grads, self.auto_encoder.classifier.model.trainable_variables))
        return np.sum(e_loss_bucket, axis=0), np.sum(d_loss_bucket, axis=0), np.sum(c_loss_bucket, axis=0)
    
    def test(self, batch_size, num_batches):
        e_loss_bucket = []
        d_loss_bucket = []
        c_loss_bucket = []
        batched_input = batch_dict(self.labelled_test_data, batch_size)

        np.random.shuffle(batched_input)
        input_dataset = batched_input[:num_batches]

        for batch_names, batch_data in input_dataset:
            e_loss, d_loss, c_loss = self.__run_batch__(batch_names, batch_data, training=False)
            e_loss_bucket.append(e_loss)
            d_loss_bucket.append(d_loss)
            c_loss_bucket.append(c_loss)
        return np.sum(e_loss_bucket, axis=0), np.sum(d_loss_bucket, axis=0), np.sum(c_loss_bucket, axis=0)
        