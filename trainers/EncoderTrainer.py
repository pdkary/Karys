import numpy as np
import tensorflow as tf

from data.wrappers.ImageDataWrapper import ImageDataWrapper
from models.EncodedClassificationModel import EncodedClassificationModel

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

class EncoderTrainer(object):
    def __init__(self,
                 encoded_classifier: EncodedClassificationModel,
                 labelled_input: ImageDataWrapper):
        self.encoded_classifier = encoded_classifier
        self.labelled_input = labelled_input

        self.labelled_train_data = self.labelled_input.get_train_dataset()
        self.labelled_test_data = self.labelled_input.get_validation_dataset()

        self.most_recent_classification = None

    def __run_batch__(self, batch_names, batch_data, training=True):
        batch_labels = [ self.labelled_input.image_labels[n] for n in batch_names]
        labels = self.encoded_classifier.classifier.label_generator.get_label_vectors_by_category_names(batch_labels)

        encoded_batch, (e_probs, e_preds) = self.encoded_classifier.encode_and_classify(batch_data, training)
        self.most_recent_encoding = list(zip(batch_data, batch_labels, encoded_batch))
        self.most_recent_classification = list(zip(batch_data, batch_labels, e_preds))
        e_loss = self.encoded_classifier.encoder.loss(labels, e_probs)
        c_loss = self.encoded_classifier.classifier.loss(labels, e_probs)
        return e_loss, c_loss

    def train(self, batch_size, num_batches):
        e_loss_bucket = []
        i_loss_bucket = []
        batched_input = batch_dict(self.labelled_train_data, batch_size)

        np.random.shuffle(batched_input)
        input_dataset = batched_input[:num_batches]

        for batch_names, batch_data in input_dataset:
            with  tf.GradientTape() as e_tape, tf.GradientTape() as c_tape:
                e_loss, c_loss = self.__run_batch__(batch_names, batch_data, training=True)
                e_loss_bucket.append(e_loss)
                i_loss_bucket.append(c_loss)

                encoder_grads = e_tape.gradient(e_loss, self.encoded_classifier.encoder.model.trainable_variables)
                classifier_grads = c_tape.gradient(c_loss, self.encoded_classifier.classifier.model.trainable_variables)
                
                self.encoded_classifier.encoder.optimizer.apply_gradients(zip(encoder_grads, self.encoded_classifier.encoder.model.trainable_variables))
                self.encoded_classifier.classifier.optimizer.apply_gradients(zip(classifier_grads, self.encoded_classifier.classifier.model.trainable_variables))
        return np.sum(e_loss_bucket, axis=0), np.sum(i_loss_bucket, axis=0)
    
    def test(self, batch_size, num_batches):
        e_loss_bucket = []
        c_loss_bucket = []
        batched_input = batch_dict(self.labelled_test_data, batch_size)

        np.random.shuffle(batched_input)
        input_dataset = batched_input[:num_batches]

        for batch_names, batch_data in input_dataset:
            e_loss, c_loss = self.__run_batch__(batch_names, batch_data, training=False)
            e_loss_bucket.append(e_loss)
            c_loss_bucket.append(c_loss)
        return np.sum(e_loss_bucket, axis=0), np.sum(c_loss_bucket, axis=0)
        