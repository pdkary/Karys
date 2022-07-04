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
        self.most_recent_inputs = None
        self.most_recent_outputs = None
    
    def get_vector_labels_like(self, like_vector, scalar_labels):
        vector_labels = np.zeros_like(like_vector)
        for i,l in enumerate(scalar_labels):
            vector_labels[i][l[0]] = 1
        return vector_labels
    
    def run_step(self, inputs, labels, training=True):
        self.most_recent_inputs = inputs
        self.most_recent_outputs = None
        with tf.GradientTape() as grad_tape:
            current_inputs = inputs
            losses = None
            for i in range(self.data_wrapper.data_config.output_length):
                current_labels = labels[:,[i]]
                outputs = self.generator.model(current_inputs,training=training)
                vector_labels = self.get_vector_labels_like(outputs, current_labels)
                loss = self.generator.loss(vector_labels, outputs)
                losses = loss if i == 0 else losses + loss
                output_scalars = np.array([[o] for o in np.argmax(outputs,axis=-1)])
                if self.most_recent_outputs is None:
                    self.most_recent_outputs = output_scalars
                else:
                    self.most_recent_outputs = np.concatenate([self.most_recent_outputs, output_scalars],axis=1)
                next_input = []
                for i,x in enumerate(current_inputs[:,1:]):
                    next_input.append(np.append(x,output_scalars[i]))
                current_inputs = np.array(next_input)
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
    
    def propogate_from_phrase(self, phrase, lookahead):
        new_out = []
        seed = [self.data_wrapper.word_index[w] for w in phrase.split(" ")]
        assert len(seed) == self.generator.model.input_shape[-1]
        current_in = np.array([seed])
        for i in range(lookahead):
            next_word_probs = self.generator.model(current_in,training=False)
            next_word = np.argmax(next_word_probs,axis=-1)[0]
            new_out.append(self.data_wrapper.index_to_word[next_word])
            current_in = np.array([np.append(current_in[0][1:], next_word)])
        return " ".join(new_out)
    
    def propogate_from_most_recent_input(self, lookahead):
        propogations = {}
        for seed in self.most_recent_inputs:
            assert len(seed) == self.generator.model.input_shape[-1]
            phrase = " ".join([self.index_to_word[w] for w in seed])
            predicted_phrase = ""
            current_in = np.array([seed])
            for i in range(lookahead):
                next_word_probs = self.generator.model(current_in,training=False)
                next_word = np.argmax(next_word_probs,axis=-1)[0]
                predicted_phrase += " " + str(self.data_wrapper.index_to_word[next_word])
                current_in = np.array([np.append(current_in[0][1:], next_word)])
            propogations[phrase] = predicted_phrase
        return propogations

        



        

