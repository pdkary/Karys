import numpy as np
import tensorflow as tf
from data.labels.CategoricalLabel import BatchedCategoricalLabel, CategoricalLabel, SequenceCategoricalLabel

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
        self.vocab_size = self.generator.output_shape[-1]
        self.train_dataset = self.data_wrapper.get_train_dataset()
        self.test_dataset = self.data_wrapper.get_validation_dataset()
        self.most_recent_inputs = None
        self.most_recent_outputs = None
        self.output_histogram = None
        
        self.gen_lbl_factory = SequenceCategoricalLabel(self.sequence_length, self.vocab_size)
        self.disc_lbl_factory = BatchedCategoricalLabel(self.discriminator.output_shape[-1])
    
    def generate_sequence(self, initial_input, scalar_labels, sequence_length, training=True, calculate_loss=True):
        current_inputs = initial_input
        outputs = []
        losses = None
        for i in range(sequence_length):
            output = self.generator.model(current_inputs,training=training)
            outputs.append(output)

            if calculate_loss:
                label_vector = self.gen_lbl_factory.get_category_sequence(scalar_labels.T[i])
                loss = self.generator.loss(label_vector, output)
                losses = loss if losses is None else losses + loss

            argmax_output = np.array([[o] for o in np.argmax(output,axis=-1)])
            next_input = np.concatenate([current_inputs[:,1:], argmax_output],axis=1)
            current_inputs = next_input
        outputs = np.array(outputs)
        outputs = np.swapaxes(outputs,0,1)
        return outputs, losses
    
    def train_on_sequence(self, initial_input, scalar_labels):
        return self.generate_sequence(initial_input,scalar_labels, self.sequence_length, training=True, calculate_loss=True)
    
    def test_on_sequence(self, initial_input, scalar_labels):
        return self.generate_sequence(initial_input, scalar_labels, self.sequence_length, training=False, calculate_loss=True)
    
    def generate_output_sequence(self, initial_input, sequence_length):
        return self.generate_sequence(initial_input, None, sequence_length, training=False, calculate_loss=False)

    def discriminate(self, labels, gen_out, training=False):
        batch_size = labels.shape[0]
        reduced_gen_out = self.extract_sentence_from_output(gen_out)
        minibatch_size = reduced_gen_out.shape[-1]

        running_gen_loss = None
        running_disc_gen_loss = None

        best_minibatch = None
        best_minibatch_score = np.inf

        succ = self.disc_lbl_factory.succ(batch_size)
        fail = self.disc_lbl_factory.fail(batch_size)
        
        labels = self.data_wrapper.scale_batch(labels)
        disc_real_out = self.discriminator.model(labels, training=training)
        disc_real_loss = self.discriminator.loss(succ, disc_real_out)

        self.output_histogram = []

        for m in range(minibatch_size):
            minibatch = reduced_gen_out[:,:,m]
                
            for sentence in minibatch:
                for word in sentence:
                    self.output_histogram.append(word)

            minibatch = self.data_wrapper.scale_batch(minibatch)
            disc_gen_out = self.discriminator.model(minibatch, training=training)

            gen_loss = self.generator.loss(succ, disc_gen_out)
            disc_gen_loss = self.discriminator.loss(fail, disc_gen_out)

            minibatch_score = np.sum(gen_loss)/(np.std(minibatch) + 1e-2)

            if minibatch_score < best_minibatch_score:
                best_minibatch_score = minibatch_score
                best_minibatch = minibatch

            running_gen_loss = gen_loss if running_gen_loss is None else running_gen_loss + gen_loss
            running_disc_gen_loss = disc_gen_loss if running_disc_gen_loss is None else running_disc_gen_loss + disc_gen_loss
        
        self.most_recent_outputs = self.data_wrapper.unscale_batch(best_minibatch)
        return running_gen_loss, running_disc_gen_loss + disc_real_loss

    def train(self, batch_size, num_batches):
        running_gen_loss, running_disc_loss = 0.0, 0.0
        dataset = self.train_dataset.shuffle(buffer_size=512).batch(batch_size).take(num_batches)
        
        for inputs, labels in list(dataset.as_numpy_iterator()):
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                self.most_recent_inputs = inputs
                gen_out, gen_loss = self.train_on_sequence(inputs, labels)
                gen_disc_loss, disc_loss = self.discriminate(labels, gen_out, training=True)
                gen_loss += gen_disc_loss

                gen_grads = gen_tape.gradient(gen_loss, self.generator.model.trainable_variables)
                disc_grads = disc_tape.gradient(disc_loss, self.discriminator.model.trainable_variables)
                self.generator.optimizer.apply_gradients(zip(gen_grads, self.generator.model.trainable_variables))
                self.discriminator.optimizer.apply_gradients(zip(disc_grads, self.discriminator.model.trainable_variables))
            running_gen_loss += np.sum(gen_loss)
            running_disc_loss += np.sum(disc_loss)
        return running_gen_loss, running_disc_loss
    
    def test(self, batch_size, num_batches):
        running_gen_loss, running_disc_loss = 0.0, 0.0
        dataset = self.test_dataset.shuffle(buffer_size=512).batch(batch_size).take(num_batches)
        for inputs, labels in list(dataset.as_numpy_iterator()):
            self.most_recent_inputs = inputs
            gen_out, gen_loss = self.test_on_sequence(inputs,labels)
            gen_disc_loss, disc_loss = self.discriminate(labels, gen_out)
            gen_loss += gen_disc_loss

            running_gen_loss += np.sum(gen_loss)
            running_disc_loss += np.sum(disc_loss)
        return running_gen_loss, running_disc_loss
    
    def propogate_from_most_recent_input(self, lookahead):
        return self.generate_output_sequence(self.most_recent_inputs, lookahead)
    
    def extract_sentence_from_output(self,gen_output,k=5):
        top_k_sentences = []
        for sentence_probs in gen_output:
            top_k_sentence = []
            for word_probs in sentence_probs:
                top_k = np.argpartition(word_probs,-k)[-k:]
                top_k_sentence.append(top_k)

            top_k_sentence = np.array(top_k_sentence)
            top_k_sentences.append(top_k_sentence)
        return np.array(top_k_sentences)
