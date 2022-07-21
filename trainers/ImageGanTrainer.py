import numpy as np
import tensorflow as tf
from data.labels.CategoricalLabel import CategoricalLabel


from data.wrappers.DataWrapper import DataWrapper
from models.ModelWrapper import ModelWrapper

def batch(ndarr, batch_size):
    N = len(ndarr)//batch_size
    return np.array(np.split(ndarr[:N*batch_size], N))

class ImageGanTrainer(object):
    def __init__(self,
                 generator: ModelWrapper,
                 discriminator: ModelWrapper,
                 generator_input: DataWrapper,
                 discriminator_input: DataWrapper):
        self.generator = generator
        self.discriminator = discriminator
        self.generator_input = generator_input
        self.discriminator_input = discriminator_input
        self.gen_train_dataset = self.generator_input.get_train_dataset()
        self.gen_test_dataset = self.generator_input.get_validation_dataset()
        self.disc_train_dataset = self.discriminator_input.get_train_dataset()
        self.disc_test_dataset = self.discriminator_input.get_validation_dataset()
        self.most_recent_gen_output = None
        self.most_recent_gen_input = None

        self.label_generator = CategoricalLabel(discriminator.output_shape[-1])

    def test_step(self,gen_input, disc_input):
        gen_out = self.generator.model(gen_input, training=True)
        self.most_recent_gen_output = gen_out
        discriminated_real_images = self.discriminator.model(disc_input, training=True)
        discriminated_genr_images = self.discriminator.model(gen_out, training=True)

        succ = self.label_generator.succ()
        fail = self.label_generator.fail()

        genr_loss = self.generator.loss(succ, discriminated_genr_images)
        disc_real_loss = self.discriminator.loss(succ, discriminated_real_images)
        disc_genr_loss = self.discriminator.loss(fail, discriminated_genr_images)

        disc_loss = disc_genr_loss + disc_real_loss
        return genr_loss, disc_loss

    def train(self, batch_size, num_batches) -> np.float32:
        gen_loss_bucket = []
        disc_loss_bucket = []

        batched_gen_input = batch(self.gen_train_dataset, batch_size)
        batched_disc_input = batch(self.disc_train_dataset, batch_size)
        np.random.shuffle(batched_gen_input)
        np.random.shuffle(batched_disc_input)

        gen_input_dataset = batched_gen_input[:num_batches]
        disc_input_dataset = batched_disc_input[:num_batches]

        for gen_input, disc_input in zip(gen_input_dataset, disc_input_dataset):
            with tf.GradientTape() as genr_tape, tf.GradientTape() as disc_tape:
                gen_loss, disc_loss = self.test_step(gen_input, disc_input)
                gen_loss_bucket.append(gen_loss)
                disc_loss_bucket.append(disc_loss)
                
                gen_grads = genr_tape.gradient(gen_loss, self.generator.model.trainable_variables)
                disc_grads = disc_tape.gradient(disc_loss, self.discriminator.model.trainable_variables)
                self.generator.optimizer.apply_gradients(zip(gen_grads, self.generator.model.trainable_variables))
                self.discriminator.optimizer.apply_gradients(zip(disc_grads, self.discriminator.model.trainable_variables))
        return np.sum(np.mean(gen_loss_bucket,axis=0)), np.sum(np.mean(disc_loss_bucket,axis=0))

    def test(self, batch_size, num_batches) -> np.float32:
        gen_loss_bucket = []
        disc_loss_bucket = []

        batched_gen_input = batch(self.gen_test_dataset, batch_size)
        batched_disc_input = batch(self.disc_test_dataset, batch_size)
        np.random.shuffle(batched_gen_input)
        np.random.shuffle(batched_disc_input)

        gen_input_dataset = batched_gen_input[:num_batches]
        disc_input_dataset = batched_disc_input[:num_batches]

        for (noise, image) in zip(gen_input_dataset, disc_input_dataset):
            gen_loss, disc_loss = self.test_step(noise,image)
            gen_loss_bucket.append(gen_loss)
            disc_loss_bucket.append(disc_loss)
        return np.sum(np.mean(gen_loss_bucket,axis=0)), np.sum(np.mean(disc_loss_bucket,axis=0))
        



        

