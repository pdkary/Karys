import numpy as np
import tensorflow as tf

from data.wrappers.DataWrapper import DataWrapper
from models.ClassificationModel import ClassificationModel

def batch(ndarr, batch_size):
    N = len(ndarr)//batch_size
    return np.array(np.split(ndarr[:N*batch_size], N))

class ClassificationTrainer(object):
    def __init__(self,
                 classifier: ClassificationModel,
                 target_input: DataWrapper,
                 noise_input: DataWrapper):
        self.classifier = classifier
        self.target_input = target_input
        self.noise_input = noise_input

        self.target_train_data = self.target_input.get_train_dataset()
        self.target_test_data = self.target_input.get_validation_dataset()
        self.noise_train_data = self.noise_input.get_train_dataset()
        self.noise_test_data = self.noise_input.get_validation_dataset()

        self.most_recent_target_output = None
        self.most_recent_noise_output = None

    def __train__(self, dataset, batch_function, batch_size, num_batches, training=True):
        loss_bucket = []
        batched_input = batch(dataset, batch_size)
        np.random.shuffle(batched_input)
        input_dataset = batched_input[:num_batches]

        for i in input_dataset:
            with tf.GradientTape() as grad_tape:
                loss = batch_function(i,training)
                loss_bucket.append(loss)
                
                classifier_grads = grad_tape.gradient(loss, self.classifier.model.trainable_variables)
                self.classifier.optimizer.apply_gradients(zip(classifier_grads, self.classifier.model.trainable_variables))
        return np.sum(loss_bucket, axis=0)

    def __run_batch__(self, batch, labels, training=True):
        _, class_probs, predicted_labels = self.classifier.classify(batch,training)
        self.most_recent_target_output = list(zip(class_probs, predicted_labels))
        return self.classifier.loss(labels, class_probs)
    
    def train_target_batch(self, target_batch, training=True):
        batch_size = target_batch.shape[0]
        target_label = self.classifier.label_generator.get_single_category(0, batch_size)
        return self.__run_batch__(target_batch, target_label, training)
    
    def train_noise_batch(self, noise_batch, training=True):
        batch_size = noise_batch.shape[0]
        noise_label = self.classifier.label_generator.get_single_category(2, batch_size)
        return self.__run_batch__(noise_batch, noise_label, training)
    
    def __test__(self, dataset, batch_function, batch_size, num_batches, training=True):
        loss_bucket = []
        batched_input = batch(dataset, batch_size)
        np.random.shuffle(batched_input)

        input_dataset = batched_input[:num_batches]
        for i in input_dataset:
            loss = batch_function(i,training)
            loss_bucket.append(loss)
        return np.sum(loss_bucket, axis=0)

    def train_target(self, batch_size, num_batches):
        return self.__train__(self.target_train_data, self.train_target_batch, batch_size, num_batches)

    def test_target(self, batch_size, num_batches):
        return self.__test__(self.target_test_data, self.train_target_batch, batch_size, num_batches, training=False)

    def train_noise(self, batch_size, num_batches):
        return self.__train__(self.noise_train_data, self.train_noise_batch, batch_size, num_batches)

    def test_noise(self, batch_size, num_batches):
        return self.__test__(self.noise_test_data, self.train_noise_batch, batch_size, num_batches, training=False)
        