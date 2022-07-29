import numpy as np
import tensorflow as tf

from data.wrappers.ImageDataWrapper import ImageDataWrapper
from models.ClassificationModel import ClassificationModel
from models.EncoderModel import EncoderModel

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
                 encoder: EncoderModel,
                 encoded_classifier: ClassificationModel,
                 image_classifier: ClassificationModel,
                 labelled_input: ImageDataWrapper):
        self.encoder = encoder
        self.encoded_classifier = encoded_classifier
        self.image_classifier = image_classifier
        self.labelled_input = labelled_input

        self.labelled_train_data = self.labelled_input.get_train_dataset()
        self.labelled_test_data = self.labelled_input.get_validation_dataset()

        self.most_recent_endcoded_classification = None
        self.most_recent_classification = None

    def __run_batch__(self, batch_names, batch_data, training=True):
        batch_labels = [ self.labelled_input.image_labels[n] for n in batch_names]
        labels = self.encoded_classifier.label_generator.get_label_vectors_by_names(batch_labels).T

        _, encoded_batch = self.encoder.encode(batch_data, training=training)
        _, e_probs, e_preds = self.encoded_classifier.classify(encoded_batch, training)
        _, c_probs, c_preds = self.image_classifier.classify(batch_data, training)

        self.most_recent_encoding = list(zip(batch_data, batch_labels, encoded_batch))
        self.most_recent_endcoded_classification = list(zip(batch_data, batch_labels, e_preds))
        self.most_recent_classification = list(zip(batch_data, batch_labels, c_preds))

        encoder_loss = self.encoder.loss(c_probs, e_probs) + self.encoder.loss(labels, e_probs)
        e_classifier_loss = self.image_classifier.loss(labels, e_probs)
        i_classifier_loss = self.image_classifier.loss(labels, c_probs)
        return encoder_loss, e_classifier_loss, i_classifier_loss

    def train(self, batch_size, num_batches):
        e_loss_bucket = []
        ec_loss_bucket = []
        ic_loss_bucket = []
        batched_input = batch_dict(self.labelled_train_data, batch_size)

        np.random.shuffle(batched_input)
        input_dataset = batched_input[:num_batches]

        for batch_names, batch_data in input_dataset:
            with  tf.GradientTape() as encoder_tape, tf.GradientTape() as e_classifier_tape, tf.GradientTape() as i_classifier_tape:
                e_loss, ec_loss, ic_loss = self.__run_batch__(batch_names, batch_data, training=True)
                e_loss_bucket.append(e_loss)
                ec_loss_bucket.append(ec_loss)
                ic_loss_bucket.append(ic_loss)

                i_classifier_grads = i_classifier_tape.gradient(ic_loss, self.image_classifier.model.trainable_variables)
                e_classifier_grads = e_classifier_tape.gradient(ec_loss, self.encoded_classifier.model.trainable_variables)
                encoder_grads = encoder_tape.gradient(e_loss, self.encoder.model.trainable_variables)
                
                self.image_classifier.optimizer.apply_gradients(zip(i_classifier_grads, self.image_classifier.model.trainable_variables))
                self.encoded_classifier.optimizer.apply_gradients(zip(e_classifier_grads, self.encoded_classifier.model.trainable_variables))
                self.encoder.optimizer.apply_gradients(zip(encoder_grads, self.encoder.model.trainable_variables))
        return np.sum(e_loss_bucket, axis=0), np.sum(ec_loss_bucket, axis=0), np.sum(ic_loss_bucket, axis=0)
    
    def test(self, batch_size, num_batches):
        e_loss_bucket = []
        ec_loss_bucket = []
        ic_loss_bucket = []
        batched_input = batch_dict(self.labelled_test_data, batch_size)

        np.random.shuffle(batched_input)
        input_dataset = batched_input[:num_batches]

        for batch_names, batch_data in input_dataset:
            e_loss, ec_loss, ic_loss = self.__run_batch__(batch_names, batch_data, training=False)
            e_loss_bucket.append(e_loss)
            ec_loss_bucket.append(ec_loss)
            ic_loss_bucket.append(ic_loss)
        return np.sum(e_loss_bucket, axis=0), np.sum(ec_loss_bucket, axis=0), np.sum(ic_loss_bucket, axis=0)
        