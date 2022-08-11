import numpy as np
import tensorflow as tf

from data.wrappers.ImageDataWrapper import ImageDataWrapper
from models.ClassificationModel import ClassificationModel

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

class ClassificationTrainer(object):
    def __init__(self,
                 classifier: ClassificationModel,
                 labelled_input: ImageDataWrapper):
        self.classifier = classifier
        self.labelled_input = labelled_input

        self.labelled_train_data = self.labelled_input.get_train_dataset()
        self.labelled_test_data = self.labelled_input.get_validation_dataset()

        self.most_recent_output = None

    def __run_batch__(self, batch_names, batch_data, training=True):
        batch_labels = [ self.labelled_input.image_labels[n] for n in batch_names]
        vector_labels = self.classifier.label_generator.get_label_vectors_by_category_names(batch_labels)

        probs, preds = self.classifier.classify(batch_data, training)
        self.most_recent_output = list(zip(batch_data, batch_labels, preds))
        return self.classifier.loss(vector_labels, probs)

    def train(self, batch_size, num_batches):
        loss_bucket = []
        batched_input = batch_dict(self.labelled_train_data, batch_size)

        np.random.shuffle(batched_input)
        input_dataset = batched_input[:num_batches]

        for batch_names, batch_data in input_dataset:
            with tf.GradientTape() as grad_tape:
                loss = self.__run_batch__(batch_names, batch_data, training=True)
                loss_bucket.append(loss)
                
                classifier_grads = grad_tape.gradient(loss, self.classifier.model.trainable_variables)
                self.classifier.optimizer.apply_gradients(zip(classifier_grads, self.classifier.model.trainable_variables))
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
        