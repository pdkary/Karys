import numpy as np
import tensorflow as tf

from data.configs.ImageDataConfig import ImageDataConfig
from data.loaders import ImageDataLoader
from data.wrappers.RandomDataWrapper import RandomDataWrapper
from models.bases.GanModelBase import GanModelBase

class GanTrainer(object):
    def __init__(self,
                 generator: GanModelBase,
                 discriminator: GanModelBase,
                 generator_input: RandomDataWrapper,
                 image_config: ImageDataConfig,
                 content_source: str):
        self.generator = generator
        self.discriminator = discriminator
        self.generator_input = generator_input
        self.image_config = image_config
        self.image_set = ImageDataLoader.load_images(content_source, image_config)
        self.most_recent_gen = None
        self.train_dataset = None
        self.test_dataset = None

    def load_datasets(self):
        train_length = int(len(self.image_set)*self.image_config.train_test_ratio)
        train_imgs = self.image_set[0:train_length]
        test_imgs = self.image_set[train_length:]
        self.train_dataset = ImageDataLoader.load_dataset(train_imgs, self.image_config)
        self.test_dataset = ImageDataLoader.load_dataset(test_imgs, self.image_config)
    
    def get_train_dataset(self):
        if self.train_dataset is None:
            self.load_datasets()
        return self.train_dataset

    def get_test_dataset(self):
        if self.test_dataset is None:
            self.load_datasets()
        return self.test_dataset
    
    def test_step(self,noise,image):
        generated_images = self.generator.model(noise, training=True)
        self.most_recent_gen = generated_images
        discriminated_real_images = self.discriminator.model(image, training=True)
        discriminated_genr_images = self.discriminator.model(generated_images, training=True)

        genr_loss = self.generator.loss(tf.ones_like(discriminated_genr_images),discriminated_genr_images)
        disc_real_loss = self.discriminator.loss(tf.ones_like(discriminated_real_images),discriminated_real_images)
        disc_genr_loss = self.discriminator.loss(tf.zeros_like(discriminated_genr_images),discriminated_genr_images)
        disc_loss = disc_genr_loss + disc_real_loss
        return genr_loss, disc_loss

    def train(self, batch_size, num_batches) -> np.float32:
        running_gen_loss = None
        running_disc_loss = None
        noise_dataset = self.generator_input.get_dataset().shuffle(buffer_size=512).batch(batch_size).take(num_batches)
        image_dataset = self.get_train_dataset().shuffle(buffer_size=512).batch(batch_size).take(num_batches)
        with tf.GradientTape() as genr_tape, tf.GradientTape() as disc_tape:
            for noise,image in zip(list(noise_dataset.as_numpy_iterator()),list(image_dataset.as_numpy_iterator())):
                if noise.shape[0] == batch_size and image.shape[0] == batch_size:
                        genr_loss,disc_loss = self.test_step(noise,image)
                        running_gen_loss = genr_loss if running_gen_loss is None else running_gen_loss + genr_loss
                        running_disc_loss = disc_loss if running_disc_loss is None else running_disc_loss + disc_loss
            genr_grads = genr_tape.gradient(running_gen_loss, self.generator.model.trainable_variables)
            disc_grads = disc_tape.gradient(running_disc_loss, self.discriminator.model.trainable_variables)
            self.generator.optimizer.apply_gradients(zip(genr_grads, self.generator.model.trainable_variables))
            self.discriminator.optimizer.apply_gradients(zip(disc_grads, self.discriminator.model.trainable_variables))
        return np.sum(running_gen_loss), np.sum(running_disc_loss)

    def test(self, batch_size, num_batches) -> np.float32:
        running_gen_loss = 0.0
        running_disc_loss = 0.0
        noise_dataset = self.generator_input.get_dataset().shuffle(buffer_size=512).batch(batch_size).take(num_batches)
        image_dataset = self.get_test_dataset().shuffle(buffer_size=512).batch(batch_size).take(num_batches)

        for noise,image in zip(list(noise_dataset.as_numpy_iterator()),list(image_dataset.as_numpy_iterator())):
            if noise.shape[0] == batch_size and image.shape[0] == batch_size:
                genr_loss,disc_loss = self.test_step(noise,image)
                running_gen_loss += np.sum(genr_loss)
                running_disc_loss += np.sum(disc_loss)
        return running_gen_loss, running_disc_loss



        

