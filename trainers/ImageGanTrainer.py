import numpy as np
import tensorflow as tf


from data.wrappers.DataWrapper import DataWrapper
from models.ModelWrapper import ModelWrapper

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
        self.most_recent_gen_output = None
        self.most_recent_gen_input = None
        self.gen_train_dataset = None
        self.gen_test_dataset = None
        self.disc_train_dataset = None
        self.disc_test_dataset = None

    def load_datasets(self):
        self.gen_train_dataset = self.generator_input.get_train_dataset()
        self.gen_test_dataset = self.generator_input.get_test_dataset()
        self.disc_train_dataset = self.discriminator_input.get_train_dataset()
        self.disc_test_dataset = self.discriminator_input.get_test_dataset()
    
    def test_step(self,gen_input, gen_labels, disc_input, disc_labels):
        gen_out = self.generator.model(gen_input, training=True)
        self.most_recent_gen_output = gen_out
        discriminated_real_images = self.discriminator.model(disc_input, training=True)
        discriminated_genr_images = self.discriminator.model(gen_out, training=True)

        genr_loss = self.generator.loss(disc_labels,discriminated_genr_images)
        disc_real_loss = self.discriminator.loss(disc_labels, discriminated_real_images)
        disc_genr_loss = self.discriminator.loss(tf.zeros_like(disc_labels),discriminated_genr_images)
        disc_loss = disc_genr_loss + disc_real_loss
        return genr_loss, disc_loss

    def train(self, batch_size, num_batches) -> np.float32:
        gen_loss_bucket = []
        disc_loss_bucket = []
        gen_input_dataset = self.gen_train_dataset.shuffle(buffer_size=512).batch(batch_size).take(num_batches)
        image_dataset = self.disc_train_dataset.shuffle(buffer_size=512).batch(batch_size).take(num_batches)

        with tf.GradientTape() as genr_tape, tf.GradientTape() as disc_tape:
            for (gen_input, gen_labels),(disc_input, disc_labels) in zip(list(gen_input_dataset.as_numpy_iterator()),list(image_dataset.as_numpy_iterator())):
                if gen_input.shape[0] == batch_size and disc_input.shape[0] == batch_size:
                    genr_loss,disc_loss = self.test_step(gen_input, gen_labels, disc_input, disc_labels)
                    gen_loss_bucket.append(genr_loss)
                    disc_loss_bucket.append(disc_loss)
            gen_loss = tf.reduce_mean(gen_loss_bucket,axis=0) / (tf.math.reduce_std(gen_loss_bucket) + 1e-2)
            disc_loss = tf.reduce_mean(disc_loss_bucket,axis=0) / (tf.math.reduce_std(disc_loss_bucket) + 1e-2)
            genr_grads = genr_tape.gradient(gen_loss, self.generator.model.trainable_variables)
            disc_grads = disc_tape.gradient(disc_loss, self.discriminator.model.trainable_variables)
            self.generator.optimizer.apply_gradients(zip(genr_grads, self.generator.model.trainable_variables))
            self.discriminator.optimizer.apply_gradients(zip(disc_grads, self.discriminator.model.trainable_variables))
        return np.sum(np.mean(gen_loss_bucket,axis=0)), np.sum(np.mean(disc_loss_bucket,axis=0))

    def test(self, batch_size, num_batches) -> np.float32:
        running_gen_loss = 0.0
        running_disc_loss = 0.0
        noise_dataset = self.gen_test_dataset.shuffle(buffer_size=512).batch(batch_size).take(num_batches)
        image_dataset = self.disc_test_dataset.shuffle(buffer_size=512).batch(batch_size).take(num_batches)

        for (noise, noise_labels),(image, image_features) in zip(list(noise_dataset.as_numpy_iterator()),list(image_dataset.as_numpy_iterator())):
            if noise.shape[0] == batch_size and image.shape[0] == batch_size:
                genr_loss,disc_loss = self.test_step(noise,noise_labels,image,image_features)
                running_gen_loss += np.sum(genr_loss)
                running_disc_loss += np.sum(disc_loss)
        return running_gen_loss, running_disc_loss
        



        

