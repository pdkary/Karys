import numpy as np
import tensorflow as tf
from data.wrappers.DataWrapper import DataWrapper

from data.wrappers.TextDataWrapper import TextDataWrapper
from models.ModelWrapper import ModelWrapper

class TextTrainer(object):
    def __init__(self,
                 generator: ModelWrapper,
                 discriminator: ModelWrapper,
                 data_wrapper: TextDataWrapper):
        self.generator: ModelWrapper = generator
        self.discriminator: ModelWrapper = discriminator
        self.data_wrapper: TextDataWrapper = data_wrapper
        self.sequence_length = self.discriminator.input_shape[-1]
        self.train_dataset = self.data_wrapper.get_train_dataset()
        self.test_dataset = self.data_wrapper.get_test_dataset()
        self.most_recent_inputs = None
        self.most_recent_outputs = None
    
    def get_vector_labels_like(self, like_vector, scalar_labels):
        vector_labels = np.zeros_like(like_vector)
        for i,l in enumerate(scalar_labels):
            vector_labels[i][l] = 1
        return vector_labels
    
    def generate_sequence(self, initial_input, scalar_labels, sequence_length, training=True, calculate_loss=True):
        current_inputs = initial_input
        outputs = []
        losses = None
        for i in range(sequence_length):
            output = self.generator.model(current_inputs,training=training)
            outputs.append(output)

            if calculate_loss:
                label_vector = self.get_vector_labels_like(output, scalar_labels.T[i])
                loss = self.generator.loss(label_vector,output)
                losses = loss if losses is None else losses + loss

            argmax_output = np.array([[o] for o in np.argmax(output,axis=-1)])
            next_input = np.concatenate([current_inputs[:,1:], argmax_output],axis=1)
            current_inputs = next_input
        outputs = np.array(outputs)
        return outputs, losses
    
    def train_on_sequence(self, initial_input, scalar_labels):
        return self.generate_sequence(initial_input,scalar_labels, self.sequence_length, training=True, calculate_loss=True)
    
    def test_on_sequence(self, initial_input, scalar_labels):
        return self.generate_sequence(initial_input, scalar_labels, self.sequence_length, training=False, calculate_loss=True)
    
    def generate_output_sequence(self, initial_input, sequence_length):
        return self.generate_sequence(initial_input, None, sequence_length, training=False, calculate_loss=False)

    def train(self, batch_size, num_batches):
        running_loss = 0.0
        dataset = self.train_dataset.shuffle(buffer_size=512).batch(batch_size).take(num_batches)
        for inputs, labels in list(dataset.as_numpy_iterator()):
            with tf.GradientTape() as gen_tape:
                self.most_recent_inputs = inputs
                outputs, losses = self.train_on_sequence(inputs, labels)
                reduced_outputs = np.argmax(outputs,axis=-1)
                self.most_recent_outputs = reduced_outputs
                grads = gen_tape.gradient(losses, self.generator.model.trainable_variables)
                self.generator.optimizer.apply_gradients(zip(grads, self.generator.model.trainable_variables))
        running_loss += np.sum(losses)
        return running_loss
    
    def test(self, batch_size, num_batches):
        running_loss = 0.0
        dataset = self.test_dataset.shuffle(buffer_size=512).batch(batch_size).take(num_batches)
        for inputs, labels in list(dataset.as_numpy_iterator()):
            self.most_recent_inputs = inputs
            outputs, losses = self.test_on_sequence(inputs,labels)
            running_loss += np.sum(losses)
        return running_loss
    
    def propogate_from_most_recent_input(self, lookahead):
        return self.generate_output_sequence(self.most_recent_inputs, lookahead)

        



        

