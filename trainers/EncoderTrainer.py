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
                 classifier: ClassificationModel,
                 labelled_input: ImageDataWrapper):
        self.encoder = encoder
        self.classifier = classifier
        self.labelled_input = labelled_input

        self.labelled_train_data = self.labelled_input.get_train_dataset()
        self.labelled_test_data = self.labelled_input.get_validation_dataset()

        self.most_recent_classification = None

    def __run_batch__(self, batch_names, batch_data, training=True):
        label_func = lambda x : self.classifier.label_generator.get_single_category(self.classifier.label_to_id[x])

        _, encoded_batch = self.encoder.encode(batch_data, training=training)
        
        batch_labels = [ self.labelled_input.image_labels[n] for n in batch_names]
        self.most_recent_encoding = list(zip(batch_data, encoded_batch, batch_labels))
        
        labels = [label_func(l) for l in batch_labels]
        _, probs, preds = self.classifier.classify(encoded_batch, training)

        self.most_recent_classification = list(zip(batch_data, batch_labels, preds))
        return self.classifier.loss(labels, probs), self.encoder.loss(labels, probs)

    def train(self, batch_size, num_batches):
        loss_bucket = []
        batched_input = batch_dict(self.labelled_train_data, batch_size)

        np.random.shuffle(batched_input)
        input_dataset = batched_input[:num_batches]

        for batch_names, batch_data in input_dataset:
            with tf.GradientTape() as classifier_tape, tf.GradientTape() as encoder_tape:
                c_loss, e_loss = self.__run_batch__(batch_names, batch_data, training=True)
                loss_bucket.append(c_loss + e_loss)
                
                classifier_grads = classifier_tape.gradient(c_loss, self.classifier.model.trainable_variables)
                encoder_grads = encoder_tape.gradient(e_loss, self.encoder.model.trainable_variables)
                self.classifier.optimizer.apply_gradients(zip(classifier_grads, self.classifier.model.trainable_variables))
                self.encoder.optimizer.apply_gradients(zip(encoder_grads, self.encoder.model.trainable_variables))
        return np.sum(loss_bucket, axis=0)
    
    def test(self, batch_size, num_batches):
        loss_bucket = []
        batched_input = batch_dict(self.labelled_test_data, batch_size)

        np.random.shuffle(batched_input)
        input_dataset = batched_input[:num_batches]

        for batch_names, batch_data in input_dataset:
            loss = self.__run_batch__(batch_names, batch_data, training=False)
            loss_bucket.append(loss)
        return np.sum(loss_bucket, axis=0)
        