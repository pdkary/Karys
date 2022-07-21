import numpy as np
import tensorflow as tf
from data.labels.CategoricalLabel import CategoricalLabel


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

        self.label_generator = CategoricalLabel(classifier.output_shape[-1])
        self.most_recent_target_output = None
        self.most_recent_noise_output = None

    def test_step(self, target_input, noise_input, training=True):
        classified_target = self.classifier.model(target_input, training=training)
        classified_noise = self.classifier.model(noise_input, training=training)

        self.most_recent_target_output = list(zip(target_input, classified_target))
        self.most_recent_noise_output = list(zip(noise_input, classified_noise))

        succ = self.label_generator.succ()
        fail = self.label_generator.fail()

        target_loss = self.classifier.loss(succ, classified_target)
        noise_loss = self.classifier.loss(fail, classified_noise)
        return target_loss, noise_loss

    def train(self, batch_size, num_batches) -> np.float32:
        target_loss_bucket, noise_loss_bucket = [], []

        batched_target_input = batch(self.target_train_data, batch_size)
        batched_noise_input = batch(self.noise_train_data, batch_size)

        np.random.shuffle(batched_target_input)
        np.random.shuffle(batched_noise_input)

        target_input_dataset = batched_target_input[:num_batches]
        noise_input_dataset = batched_noise_input[:num_batches]

        for target_input, noise_input in zip(target_input_dataset, noise_input_dataset):
            with tf.GradientTape() as grad_tape:
                target_loss, noise_loss = self.test_step(target_input, noise_input)
                target_loss_bucket.append(target_loss)
                noise_loss_bucket.append(noise_loss)
                
                classifier_grads = grad_tape.gradient(target_loss + noise_loss, self.classifier.model.trainable_variables)
                self.classifier.optimizer.apply_gradients(zip(classifier_grads, self.classifier.model.trainable_variables))
        return np.sum(np.mean(target_loss_bucket,axis=0)), np.sum(np.mean(noise_loss_bucket,axis=0))

    def test(self, batch_size, num_batches) -> np.float32:
        target_loss_bucket, noise_loss_bucket = [], []

        batched_target_input = batch(self.target_test_data, batch_size)
        batched_noise_input = batch(self.noise_test_data, batch_size)

        np.random.shuffle(batched_target_input)
        np.random.shuffle(batched_noise_input)
        target_input_dataset = batched_target_input[:num_batches]
        noise_input_dataset = batched_noise_input[:num_batches]

        for target_input, noise_input in zip(target_input_dataset, noise_input_dataset):
            target_loss, noise_loss = self.test_step(target_input, noise_input, training=False)
            target_loss_bucket.append(target_loss)
            noise_loss_bucket.append(noise_loss)
        return np.sum(np.mean(target_loss_bucket,axis=0)), np.sum(np.mean(noise_loss_bucket,axis=0))
        