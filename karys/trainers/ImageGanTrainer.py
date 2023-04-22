import numpy as np
import tensorflow as tf
from karys.data.ImageDataLoader import ImageDataLoader
from keras.models import Model
from keras.optimizers import Optimizer
from keras.losses import Loss 
import keras.backend as K
from matplotlib import pyplot as plt
from PIL import Image

def scale_to_255(img):
    img_min = np.min(img)
    img_max = np.max(img)
    return 255*(img - img_min)/(img_max - img_min + 1e-8)

class ImageGanTrainer(object):
    def __init__(self,
                 generator: Model,
                 discriminator: Model,
                 gen_optimizer: Optimizer,
                 gen_loss: Loss,
                 disc_optimizer: Optimizer,
                 disc_loss: Loss,
                 style_loss: Loss,
                 labelled_image_input: ImageDataLoader):
        self.generator = generator
        self.discriminator = discriminator
        self.labelled_image_input: ImageDataLoader = labelled_image_input
        self.image_shape = discriminator.input_shape[-3:-1]

        self.gen_optimizer: Optimizer = gen_optimizer
        self.gen_loss: Loss = gen_loss
        self.disc_optimizer: Optimizer = disc_optimizer
        self.disc_loss: Loss = disc_loss
        self.style_loss: Loss = style_loss
        
        self.generator.compile(optimizer=gen_optimizer, loss=gen_loss)
        self.discriminator.compile(optimizer=disc_optimizer, loss=disc_loss)

        self.most_recent_gen_output = None
        self.most_recent_gen_input = None

    def __run_batch__(self, noise_batch, image_batch, image_names, training=False):
        generated_images = self.generator(noise_batch, training=training)

        flag = self.labelled_image_input.label_set[-1]
        image_categories = [name.split('_')[0] for name in image_names]
        flagged_categories = [flag for name in image_names]
        real_labels = self.labelled_image_input.get_label_vectors_by_names(image_categories)
        fake_labels = self.labelled_image_input.get_label_vectors_by_names(flagged_categories)
        
        gen_features, gen_classification_probs = self.discriminator(generated_images, training=training)
        argmax_gen_class = np.argmax(gen_classification_probs, axis=1)
        pred_gen_labels = self.labelled_image_input.get_label_vectors_by_ids(argmax_gen_class)
        
        real_features, real_classification_probs = self.discriminator(image_batch, training=training)
        argmax_real_class = np.argmax(real_classification_probs, axis=1)
        pred_real_labels = self.labelled_image_input.get_label_vectors_by_ids(argmax_real_class)

        style_axis = [1]
        real_features_std, real_features_mean = K.std(real_features, axis=style_axis, keepdims=True), K.mean(real_features, axis=style_axis, keepdims=True)
        gen_features_std, gen_features_mean = K.std(gen_features, axis=style_axis, keepdims=True), K.mean(gen_features, axis=style_axis, keepdims=True)
        
        real_image_std, real_image_mean = K.std(K.constant(image_batch), axis=[1,2,3], keepdims=True), K.mean(K.constant(image_batch), axis=[1,2,3], keepdims=True)
        gen_image_std, gen_image_mean = K.std(generated_images, axis=[1,2,3], keepdims=True), K.mean(generated_images, axis=[1,2,3], keepdims=True)
        
        gen_style_loss = self.style_loss(real_features_std, gen_features_std)
        gen_style_loss += self.style_loss(real_features_mean, gen_features_mean)
        gen_style_loss += self.style_loss(real_image_std, gen_image_std)
        gen_style_loss += self.style_loss(real_image_mean, gen_image_mean)
        
        disc_style_loss = self.style_loss(tf.ones_like(real_features_std), real_features_std)
        disc_style_loss += self.style_loss(tf.ones_like(real_features_mean), real_features_mean)

        genr_loss = self.gen_loss(0.5*(real_labels + tf.ones_like(real_labels) - fake_labels), gen_classification_probs)
        # genr_loss += 0.5*self.gen_loss(0.5*tf.ones_like(real_labels), gen_classification_probs)

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
        
        batched_image_input = self.labelled_image_input.get_sized_training_batch_set(num_batches, batch_size, self.image_shape)
        batched_noise_input = np.random.normal(0,1.0,size=(num_batches, batch_size, self.generator.input_shape[-1]))

        for noise_batch, (batch_labels, image_batch) in zip(batched_noise_input, batched_image_input):
            with tf.GradientTape() as genr_tape, tf.GradientTape() as disc_tape:
                gen_loss, disc_loss, gen_style_loss, disc_style_loss = self.__run_batch__(noise_batch, image_batch, batch_labels)
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

        batched_image_input = self.labelled_image_input.get_sized_validation_batch_set(num_batches, batch_size, self.image_shape)
        batched_noise_input = np.random.normal(0,1.0,size=(num_batches, batch_size, self.generator.input_shape[-1]))

        for noise_batch, (image_names, image_batch) in zip(batched_noise_input, batched_image_input):
            gen_loss, disc_loss, gen_style_loss, disc_style_loss = self.__run_batch__(noise_batch, image_batch, image_names)
            gen_loss_bucket.append(gen_loss)
            disc_loss_bucket.append(disc_loss)
            gen_style_loss_bucket.append(gen_style_loss)
            disc_style_loss_bucket.append(disc_style_loss)
                
        return np.sum(np.mean(gen_loss_bucket,axis=0)), np.sum(np.mean(disc_loss_bucket,axis=0)), np.sum(np.mean(gen_style_loss_bucket,axis=0)), np.sum(np.mean(disc_style_loss_bucket,axis=0))
    
    def save_classified_real_images(self, filename, preview_rows, preview_cols, img_size=32):
        self.__save_classified_images__(filename, self.most_recent_output, preview_rows, preview_cols, img_size)
    
    
    def save_classified_gen_images(self, filename, preview_rows, preview_cols, img_size=32):
        self.__save_classified_images__(filename, self.most_recent_gen_output, preview_rows, preview_cols, img_size)

    def __save_classified_images__(self, filename, dataset, preview_rows, preview_cols, img_size = 32):
        image_shape = (img_size, img_size)
        channels = image_shape[-1]

        fig,axes = plt.subplots(preview_rows,preview_cols, figsize=image_shape)

        for row in range(preview_rows):
            for col in range(preview_cols):
                img, label, pred = dataset[row*preview_rows+col]
                pass_fail = "PASS" if np.all(pred == label) else "FAIL"
                text_label = pred + " || " + label + " || " + pass_fail
                img = scale_to_255(img)
                if channels == 1:
                    img = np.reshape(img,newshape=image_shape)
                else:
                    img = np.array(img)
                    img = Image.fromarray((img).astype(np.uint8))
                    img = img.resize(image_shape, Image.BICUBIC)
                    img = np.asarray(img)
                    
                axes[row,col].imshow(img)
                axes[row,col].set_title(text_label, fontsize=img_size//2)

        fig.savefig(filename)
        plt.close()