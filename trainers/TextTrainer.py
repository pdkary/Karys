import numpy as np
import tensorflow as tf
from data.wrappers.DataWrapper import DataWrapper

from data.wrappers.TextDataWrapper import TextDataWrapper
from models.ModelWrapper import ModelWrapper

class TextTrainer(object):
    def __init__(self,
                 model: ModelWrapper,
                 data_wrapper: TextDataWrapper):
        self.generator: ModelWrapper = model
        self.data_wrapper: TextDataWrapper = data_wrapper
        self.train_dataset = self.data_wrapper.get_train_dataset()
        self.test_dataset = self.data_wrapper.get_test_dataset()
        self.most_recent_input = None
    
    def run_step(self,inputs, labels, training=True):
        with tf.GradientTape() as grad_tape:
            self.most_recent_input = inputs
            outputs = self.generator.model(inputs,training=training)
            z = np.zeros_like(outputs)
            z[0][labels[0][0]] = 1.0
            losses = self.generator.loss(z, outputs)
            if training:
                grads = grad_tape.gradient(losses, self.generator.model.trainable_variables)
                self.generator.optimizer.apply_gradients(zip(grads, self.generator.model.trainable_variables))
        return losses

    def train(self, batch_size, num_batches):
        running_loss = 0.0
        dataset = self.train_dataset.shuffle(buffer_size=512).batch(batch_size).take(num_batches)
        for inputs, labels in list(dataset.as_numpy_iterator()):
            losses = self.run_step(inputs,labels,training=True)
            running_loss += np.sum(losses)
        return running_loss
    
    def test(self, batch_size, num_batches):
        running_loss = 0.0
        dataset = self.test_dataset.shuffle(buffer_size=512).batch(batch_size).take(num_batches)
        for inputs, labels in list(dataset.as_numpy_iterator()):
            losses = self.run_step(inputs,labels,training=True)
            running_loss += np.sum(losses)
        return running_loss
    
    def propogate_from_seed(self, seed, lookahead):
        new_out = []
        current_in = seed
        for i in range(lookahead):
            next_word_probs = self.generator.model(current_in,training=False)
            next_word = np.argmax(next_word_probs,axis=-1)[0]
            new_out.append(self.data_wrapper.index_to_word[next_word])
            current_in = np.array([np.append(current_in[0][1:], next_word)])
        return " ".join(new_out)

        



        

