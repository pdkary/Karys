import numpy as np
import tensorflow as tf
from data.labels.CategoricalLabel import BatchedCategoricalLabel


from data.wrappers.DataWrapper import DataWrapper
from models.ModelWrapper import ModelWrapper

def batch(ndarr, batch_size):
    N = len(ndarr)//batch_size
    return np.array(np.split(ndarr[:N*batch_size], N))

class ClassificationTrainer(object):
    def __init__(self,
                 classifier: ModelWrapper,
                 target_input: DataWrapper,
                 noise_input: DataWrapper):
        self.classifier = classifier
        self.target_input = target_input
        self.noise_input = noise_input

        self.target_train_data = self.target_input.get_train_dataset()
        self.target_test_data = self.target_input.get_validation_dataset()
        self.noise_train_data = self.noise_input.get_train_dataset()
        self.noise_test_data = self.noise_input.get_validation_dataset()

        ##label shape will be [a b c]^T where a hot dog has values [1 0 0] and not hot dog has [0 0 1], [0 1 0] implies unknown
        label_dict = {0:"HOT DOG", 1:"UNSURE", 2:"NOT HOT DOG"}
        self.label_generator = BatchedCategoricalLabel(3, label_dict)
        self.most_recent_target_output = None
        self.most_recent_noise_output = None
    
    def train_target_batch(self, target_batch, training=True):
        batch_size = target_batch.shape[0]
        classified_target = self.classifier.model(target_batch, training=training)
        argmax_target = np.argmax(classified_target, axis=1)

        target_class_labels = [self.label_generator.label_dict[x] for x in argmax_target]
        self.most_recent_target_output = list(zip(target_batch, target_class_labels))
        
        target_label = self.label_generator.get_single_categories(0, batch_size)
        target_loss = self.classifier.loss(target_label, classified_target)
        return target_loss
    
    def train_noise_batch(self, noise_batch, training=True):
        batch_size = noise_batch.shape[0]
        classified_noise = self.classifier.model(noise_batch, training=training)
        argmax_noise = np.argmax(classified_noise, axis=1)

        noise_class_labels = [self.label_generator.label_dict[x] for x in argmax_noise]
        self.most_recent_noise_output = list(zip(noise_batch, noise_class_labels))
        
        noise_label = self.label_generator.get_single_categories(2, batch_size)
        noise_loss = self.classifier.loss(noise_label, classified_noise)
        return noise_loss

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

    def train_target(self, batch_size, num_batches):
        return self.__train__(self.target_train_data, self.train_target_batch, batch_size, num_batches)

    def test_target(self, batch_size, num_batches):
        return self.__train__(self.target_test_data, self.train_target_batch, batch_size, num_batches, training=False)

    def train_noise(self, batch_size, num_batches):
        return self.__train__(self.noise_train_data, self.train_noise_batch, batch_size, num_batches)

    def test_noise(self, batch_size, num_batches):
        return self.__train__(self.noise_test_data, self.train_noise_batch, batch_size, num_batches, training=False)
        