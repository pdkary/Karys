import numpy as np
import tensorflow as tf

from data.wrappers.ImageDataWrapper import ImageDataWrapper
from models.EncodedClassificationModel import EncodedClassificationModel
from models.GenerationModel import GenerationModel

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
                 encoded_classifier: EncodedClassificationModel,
                 generator: GenerationModel,
                 labelled_input: ImageDataWrapper):
        self.encoded_classifier: EncodedClassificationModel = encoded_classifier
        self.generator: GenerationModel = generator
        self.labelled_input = labelled_input

        self.labelled_train_data = self.labelled_input.get_train_dataset()
        self.labelled_test_data = self.labelled_input.get_validation_dataset()

        self.most_recent_generated = None
        self.most_recent_real_encoding = None
        self.most_recent_gen_encoding = None
        self.most_recent_real_classification = None
        self.most_recent_gen_classification = None

    def __run_batch__(self, batch_names, batch_data, training=True):
        batch_labels = [ self.labelled_input.image_labels[n] for n in batch_names]
        labels = self.encoded_classifier.classifier.label_generator.get_label_vectors_by_names(batch_labels)

        encoded_batch, e_probs, e_preds = self.encoded_classifier.classify(batch_data, training)
        
        generated_batch_data = self.generator.generate(encoded_batch, training)
        encoded_g_batch, g_probs, g_preds = self.encoded_classifier.classify(generated_batch_data, training)

        self.most_recent_generated = generated_batch_data
        self.most_recent_real_encoding = list(zip(batch_data, batch_labels, encoded_batch))
        self.most_recent_gen_encoding = list(zip(generated_batch_data, batch_labels, encoded_g_batch))
        self.most_recent_real_classification = list(zip(batch_data, batch_labels, e_preds))
        self.most_recent_gen_classification = list(zip(generated_batch_data, batch_labels, g_preds))

        void_label = self.encoded_classifier.classifier.label_generator.get_label_vector_by_name("Invalid")
        void_labels = np.repeat(void_label, batch_data.shape[0],axis=0)

        generator_encoding_loss = self.generator.loss(encoded_batch, encoded_g_batch)
        generator_classification_loss = self.generator.loss(1-void_labels, g_probs)

        classifier_real_loss = self.encoded_classifier.classifier.loss(labels, e_probs)
        classifier_gen_loss = self.encoded_classifier.classifier.loss(void_labels, g_probs)

        encoder_real_std = np.std(encoded_batch, axis=0)
        encoder_gen_std = np.std(encoded_g_batch, axis=0)

        encoder_real_var_loss = self.encoded_classifier.encoder.loss(np.ones_like(encoder_real_std), encoder_real_std)
        encoder_gen_var_loss = self.encoded_classifier.encoder.loss(np.ones_like(encoder_gen_std), encoder_gen_std)

        generator_loss = generator_encoding_loss + generator_classification_loss
        classifier_loss = classifier_real_loss + classifier_gen_loss
        encoder_loss = encoder_real_var_loss + encoder_gen_var_loss + classifier_loss
        return encoder_loss, classifier_loss, generator_loss

    def train(self, batch_size, num_batches):
        e_loss_bucket = []
        c_loss_bucket = []
        g_loss_bucket = []
        batched_input = batch_dict(self.labelled_train_data, batch_size)

        np.random.shuffle(batched_input)
        input_dataset = batched_input[:num_batches]

        for batch_names, batch_data in input_dataset:
            with  tf.GradientTape() as e_tape, tf.GradientTape() as c_tape, tf.GradientTape() as g_tape:
                e_loss, c_loss, g_loss = self.__run_batch__(batch_names, batch_data, training=True)
                e_loss_bucket.append(e_loss)
                c_loss_bucket.append(c_loss)
                g_loss_bucket.append(g_loss)

                e_grads = e_tape.gradient(e_loss, self.encoded_classifier.encoder.model.trainable_variables)
                c_grads = c_tape.gradient(c_loss, self.encoded_classifier.classifier.model.trainable_variables)
                g_grads = g_tape.gradient(g_loss, self.generator.model.trainable_variables)
                
                self.encoded_classifier.encoder.optimizer.apply_gradients(zip(e_grads, self.encoded_classifier.encoder.model.trainable_variables))
                self.encoded_classifier.classifier.optimizer.apply_gradients(zip(c_grads, self.encoded_classifier.classifier.model.trainable_variables))
                self.generator.optimizer.apply_gradients(zip(g_grads, self.generator.model.trainable_variables))
        return np.sum(e_loss_bucket, axis=0), np.sum(c_loss_bucket, axis=0), np.sum(g_loss_bucket, axis=0)
    
    def test(self, batch_size, num_batches):
        e_loss_bucket = []
        c_loss_bucket = []
        g_loss_bucket = []
        batched_input = batch_dict(self.labelled_test_data, batch_size)

        np.random.shuffle(batched_input)
        input_dataset = batched_input[:num_batches]

        for batch_names, batch_data in input_dataset:
            e_loss, c_loss, g_loss = self.__run_batch__(batch_names, batch_data, training=False)
            e_loss_bucket.append(e_loss)
            c_loss_bucket.append(c_loss)
            g_loss_bucket.append(g_loss)
        return np.sum(e_loss_bucket, axis=0), np.sum(c_loss_bucket, axis=0), np.sum(g_loss_bucket, axis=0)
        