import tensorflow as tf
from keras.models import Model
from keras.optimizers import Optimizer
from keras.losses import Loss
from random import randint

from karys.data.wrappers.TextDataWrapper import TextDataWrapper

class TextRNNTrainer:
    def __init__(self,
                 rnn_model: Model,
                 optimizer: Optimizer,
                 loss: Loss,
                 text_data_wrapper: TextDataWrapper):
        self.rnn_model: Model = rnn_model
        self.optimizer: Optimizer = optimizer
        self.loss: Loss = loss
        self.text_data_wrapper: TextDataWrapper = text_data_wrapper
        self.rnn_model.compile(optimizer=self.optimizer, loss=loss)

        self.train_data = text_data_wrapper.get_train_dataset()
        self.test_data = text_data_wrapper.get_validation_dataset()
        self.most_recent_output = None
    
    def save(self, model_path):
        self.rnn_model.save(model_path)
    
    def print_most_recent_output(self):
        mro = self.most_recent_output.numpy()
        mri = self.most_recent_input.numpy()
        out_sentences = self.text_data_wrapper.translate_sentences(mro)
        in_sentences = self.text_data_wrapper.translate_sentences(mri)
        print("\t",in_sentences[0],':\n\t\t- ',out_sentences[0])
    
    def train(self, batch_size, num_batches):
        train_batches = self.train_data
        random_takes = [randint(0, len(train_batches)) for _ in range(num_batches)]
        batches = [(i,train_batches[i:i+batch_size]) for i in random_takes]

        with tf.GradientTape() as grad_tape:
            vocab_size = self.text_data_wrapper.vocab_size
            loss = 0
            for seq_i, batch in batches:
                batch_input = tf.convert_to_tensor([b[0] for b in batch], dtype=tf.int64)
                batch_output = tf.convert_to_tensor([b[1] for b in batch], dtype=tf.int64)
                
                vector_labels = tf.convert_to_tensor(tf.one_hot(batch_output, vocab_size), dtype=tf.float32)
                vector_output = self.rnn_model(batch_input)

                argmax_output = tf.argmax(vector_output, axis=-1, output_type=tf.int64)
                self.most_recent_input = batch_input
                self.most_recent_output = argmax_output
                loss += self.loss(vector_labels, vector_output)      
            grads = grad_tape.gradient(loss, self.rnn_model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.rnn_model.trainable_variables))
        return loss
    
