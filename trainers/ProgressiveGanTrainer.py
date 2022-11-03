import numpy as np
import tensorflow as tf
from data.labels.CategoricalLabel import CategoricalLabel
from keras.models import Model
from keras.optimizers import Optimizer
from keras.losses import Loss 
import tensorflow.keras.backend as K

from data.wrappers.ImageDataWrapper import ImageDataWrapper
from data.wrappers.RandomDataWrapper import RandomDataWrapper

def batch(ndarr, batch_size):
    N = len(ndarr)//batch_size
    return np.array(np.split(ndarr[:N*batch_size], N))

def batch_dict(adict, batch_size):
    N = len(adict)//batch_size
    batches = []
    for i in range(N):
        b_names = list(adict.keys())[i*batch_size:(i+1)*batch_size]
        b_vals = np.array(list(adict.values())[i*batch_size:(i+1)*batch_size])
        batches.append((b_names, b_vals))
    return batches

class ProgressiveGanTrainer(object):
    def __init__(self,
                 generator: Model,
                 discriminator: Model,
                 gen_optimizer: Optimizer,
                 gen_loss: Loss,
                 disc_optimizer: Optimizer,
                 disc_loss: Loss,
                 style_loss: Loss,
                 noise_input: RandomDataWrapper,
                 labelled_image_input: ImageDataWrapper):
        self.generator = generator
        self.discriminator = discriminator
        self.noise_input = noise_input
        self.labelled_image_input = labelled_image_input

        self.gen_optimizer: Optimizer = gen_optimizer
        self.gen_loss: Loss = gen_loss
        self.disc_optimizer: Optimizer = disc_optimizer
        self.disc_loss: Loss = disc_loss
        self.style_loss: Loss = style_loss
        
        self.generator.compile(optimizer=gen_optimizer, loss=gen_loss)
        self.discriminator.compile(optimizer=disc_optimizer, loss=disc_loss)

        self.noise_train_dataset = self.noise_input.get_train_dataset()
        self.noise_test_dataset = self.noise_input.get_validation_dataset()
        self.image_train_dataset = self.labelled_image_input.get_train_dataset()
        self.image_test_dataset = self.labelled_image_input.get_validation_dataset()
        self.most_recent_gen_output = None
        self.most_recent_gen_input = None

    def __run_batch__(self, noise_batch, image_batch, image_names, training=False):
        generated_images = self.generator(noise_batch, training=training)

        blank_count = 1
        for i in range(generated_images.shape[0]):
            gen_i = generated_images[i]
            img_min = np.min(gen_i)
            img_max = np.max(gen_i)
            img_std = np.std(gen_i)
            img_range = img_max - img_min
            if img_range == 0 or img_std == 0 or 6*img_std < img_range:
                blank_count+=1
                print("blank",end="")

        flag = self.labelled_image_input.non_present_labels[0]
        image_categories = [name.split('_')[0] for name in image_names]
        flagged_categories = [flag for name in image_names]
        real_labels = self.labelled_image_input.label_vectorizer.get_label_vectors_by_category_names(image_categories)
        fake_labels = self.labelled_image_input.label_vectorizer.get_label_vectors_by_category_names(flagged_categories)
        
        gen_features, gen_classification_probs = self.discriminator(generated_images, training=training)
        argmax_gen_class = np.argmax(gen_classification_probs, axis=1)
        pred_gen_labels = self.labelled_image_input.label_vectorizer.get_category_names_by_ids(argmax_gen_class)
        
        real_features, real_classification_probs = self.discriminator(image_batch, training=training)
        argmax_real_class = np.argmax(real_classification_probs, axis=1)
        pred_real_labels = self.labelled_image_input.label_vectorizer.get_category_names_by_ids(argmax_real_class)

        style_axis = [1]
        real_std, real_mean = K.std(real_features, axis=style_axis, keepdims=True), K.mean(real_features, axis=style_axis, keepdims=True)
        gen_std, gen_mean = K.std(gen_features, axis=style_axis, keepdims=True), K.mean(gen_features, axis=style_axis, keepdims=True)

        gen_style_loss = self.style_loss(real_std, gen_std)
        gen_style_loss += self.style_loss(real_mean, gen_mean)
        gen_style_loss *= blank_count*blank_count
        
        disc_style_loss = self.style_loss(tf.ones_like(real_std), real_std)
        disc_style_loss += self.style_loss(tf.ones_like(real_mean), real_mean)

        genr_loss = self.gen_loss(0.5*(real_labels + tf.ones_like(real_labels) - fake_labels), gen_classification_probs)
        # genr_loss += 0.5*self.gen_loss(0.5*tf.ones_like(real_labels), gen_classification_probs)
        genr_loss *= blank_count*blank_count

        disc_loss = 0.5*self.disc_loss(real_labels, real_classification_probs)
        disc_loss += 0.5*self.disc_loss(fake_labels, gen_classification_probs)

        self.most_recent_output = list(zip(image_batch, image_categories, pred_real_labels))
        self.most_recent_gen_output = list(zip(generated_images, flagged_categories, pred_gen_labels))
        return genr_loss, disc_loss, gen_style_loss, disc_style_loss

    def train(self, batch_size, num_batches) -> np.float32:
        gen_loss_bucket = []
        disc_loss_bucket = []
        gen_style_loss_bucket = []
        disc_style_loss_bucket = []

        batched_noise_input = batch(self.noise_train_dataset, batch_size)
        batched_image_input = batch_dict(self.image_train_dataset, batch_size)
        np.random.shuffle(batched_noise_input)
        np.random.shuffle(batched_image_input)

        noise_input_dataset = batched_noise_input[:num_batches]
        image_input_dataset = batched_image_input[:num_batches]

        for noise_batch, (image_names, image_batch) in zip(noise_input_dataset, image_input_dataset):
            with tf.GradientTape() as genr_tape, tf.GradientTape() as disc_tape:
                gen_loss, disc_loss, gen_style_loss, disc_style_loss = self.__run_batch__(noise_batch, image_batch, image_names)
                gen_loss_bucket.append(gen_loss)
                disc_loss_bucket.append(disc_loss)
                gen_style_loss_bucket.append(gen_style_loss)
                disc_style_loss_bucket.append(disc_style_loss)
                
                gen_grads = genr_tape.gradient(gen_loss + gen_style_loss, self.generator.trainable_variables)
                disc_grads = disc_tape.gradient(disc_loss + disc_style_loss, self.discriminator.trainable_variables)
                self.gen_optimizer.apply_gradients(zip(gen_grads, self.generator.trainable_variables))
                self.disc_optimizer.apply_gradients(zip(disc_grads, self.discriminator.trainable_variables))
        return np.sum(np.mean(gen_loss_bucket,axis=0)), np.sum(np.mean(disc_loss_bucket,axis=0)), np.sum(np.mean(gen_style_loss_bucket,axis=0)), np.sum(np.mean(disc_style_loss_bucket,axis=0))

    def test(self, batch_size, num_batches) -> np.float32:
        gen_loss_bucket = []
        disc_loss_bucket = []
        gen_style_loss_bucket = []
        disc_style_loss_bucket = []

        batched_noise_input = batch(self.noise_train_dataset, batch_size)
        batched_image_input = batch_dict(self.image_train_dataset, batch_size)
        np.random.shuffle(batched_noise_input)
        np.random.shuffle(batched_image_input)

        noise_input_dataset = batched_noise_input[:num_batches]
        image_input_dataset = batched_image_input[:num_batches]

        for noise_batch, (image_names, image_batch) in zip(noise_input_dataset, image_input_dataset):
            gen_loss, disc_loss, gen_style_loss, disc_style_loss = self.__run_batch__(noise_batch, image_batch, image_names)
            gen_loss_bucket.append(gen_loss)
            disc_loss_bucket.append(disc_loss)
            gen_style_loss_bucket.append(gen_style_loss)
            disc_style_loss_bucket.append(disc_style_loss)
                
        return np.sum(np.mean(gen_loss_bucket,axis=0)), np.sum(np.mean(disc_loss_bucket,axis=0)), np.sum(np.mean(gen_style_loss_bucket,axis=0)), np.sum(np.mean(disc_style_loss_bucket,axis=0))
        