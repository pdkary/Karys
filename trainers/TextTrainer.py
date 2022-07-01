import numpy as np
import tensorflow as tf
from data.configs.TextDataConfig import TextDataConfig
from data.wrappers.DataWrapper import DataWrapper

from data.wrappers.TextDataWrapper import TextDataWrapper
from models.ModelWrapper import ModelWrapper
from trainers.outputs.TextModelOutput import TextModelOutput

class TextTrainer(object):
    def __init__(self,
                 model: ModelWrapper,
                 data_wrapper: DataWrapper):
        self.model_base: ModelWrapper = model
        self.data_wrapper: DataWrapper = data_wrapper
        self.train_dataset = self.data_wrapper.get_train_dataset()
        self.test_dataset = self.data_wrapper.get_test_dataset()

    def train(self, batch_size, num_batches):
        running_loss = 0.0
        dataset = self.train_dataset.batch(batch_size).shuffle(buffer_size=512).take(num_batches)
        for inputs, labels in list(dataset.as_numpy_iterator()):
            with tf.GradientTape() as grad_tape:
                outputs = self.model_base.model(inputs,training=True)
                losses = self.model_base.loss(labels, outputs)
                running_loss += np.sum(losses)
                grads = grad_tape.gradient(losses, self.model_base.model.trainable_variables)
                self.model_base.optimizer.apply_gradients(zip(grads, self.model_base.model.trainable_variables))
        return running_loss
    
    def test(self, batch_size, num_batches):
        running_loss = 0.0
        output = TextModelOutput()
        dataset = self.test_dataset.batch(batch_size).shuffle(buffer_size=512).take(num_batches)
        translate = lambda x: self.text_data_wrapper.translate_sentence(x)
        for inputs, labels in list(dataset.as_numpy_iterator()):
            outputs = self.model_base.model(inputs,training=True)
            for i,o in zip(inputs,outputs):
                i_s = [translate(x) for x in i]
                o_s = [translate(x) for x in o.numpy()]
                output.add_output(i_s,o_s)
            losses = self.model_base.loss(labels, outputs)
            running_loss += np.sum(losses)
        return running_loss, output



        

