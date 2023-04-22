import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.optimizers import Optimizer
from keras.losses import Loss 
import keras.backend as K
from PIL import Image
import matplotlib.pyplot as plt

from karys.data.ImageDataLoader import ImageDataLoader

def scale_to_255(img):
    img_min = np.min(img)
    img_max = np.max(img)
    return 255*(img - img_min)/(img_max - img_min + 1e-8)

def get_all_layers(model: Model):
    out_layers = []
    for x in model.layers:
        if hasattr(x, "layers"):
            out_layers.extend(get_all_layers(x))
        else:
            out_layers.append(x)
    return out_layers

class ProgressiveDiscriminatorTrainer(object):
    def __init__(self,
                 discriminator: Model,
                 disc_optimizer: Optimizer,
                 disc_loss: Loss,
                 style_loss: Loss,
                 labelled_image_input: ImageDataLoader,
                 tensorboard_writer):
        self.discriminator = discriminator
        self.labelled_image_input = labelled_image_input
        self.image_shape = discriminator.input_shape[1:]

        self.disc_optimizer: Optimizer = disc_optimizer
        self.disc_loss: Loss = disc_loss
        self.style_loss: Loss = style_loss

        self.writer = tensorboard_writer
        self.step = 0
        
        print("Discriminator input shape: ", discriminator.input_shape)
        print("Discriminator output shape: ", discriminator.output_shape)
        self.discriminator.compile(optimizer=disc_optimizer, loss=disc_loss)
        print("Discriminator compiled: ", discriminator)
        self.most_recent_output = []

    def __run_batch__(self, img_224, image_labels, image_label_vectors, training=False):
        disc_output = self.discriminator(img_224, training=training)
        classification_probs = disc_output[0]
        probs_argmax = K.argmax(classification_probs, axis=-1).numpy()
        disc_loss = self.disc_loss(image_label_vectors, classification_probs)
        self.most_recent_output = [(img_224, image_labels, probs_argmax)]

        
        real_features = disc_output[-1]
        with self.writer.as_default():
            for lbl, real_feature in zip(image_labels, real_features):
                tf.summary.histogram(f"label_features/{lbl}", real_feature.numpy(), step=self.step)
        
        for extra_out in disc_output[1:-1]:
            self.most_recent_output.append((extra_out, image_labels, probs_argmax))
                
        return disc_loss
    
    def train_step(self, image_labels, img_lbl_vectors, img_224):
        with tf.GradientTape() as disc_tape:
            # print("IN TRAIN STEP: ", np.array(img_224).shape, image_labels, np.array(img_lbl_vectors).shape)
            # print("VECTORS: ", img_lbl_vectors)
            disc_loss = self.__run_batch__(img_224, image_labels, img_lbl_vectors, training=True)
            # print("DISC LOSS: ", disc_loss)
            # print(len(self.discriminator.trainable_variables),[x.name for x in self.discriminator.trainable_variables])
            disc_grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
            
        self.log_training_tensorboard_data(disc_loss)
        self.disc_optimizer.apply_gradients(zip(disc_grads, self.discriminator.trainable_variables))
        return disc_loss
    
    def train(self, batch_size, num_batches) -> tf.float32:
        disc_loss_bucket = []
        batched_image_input = self.labelled_image_input.get_sized_training_batch_set(num_batches, batch_size, (224,224))

        for image_labels, img_lbl_vectors, flag_vectors, img_224 in batched_image_input:
            disc_loss = self.train_step(image_labels, img_lbl_vectors, img_224)
            disc_loss_bucket.append(disc_loss)
        return np.sum(np.mean(disc_loss_bucket,axis=0))

    def test(self, batch_size, num_batches):
        disc_loss_bucket = []
        batched_image_input = self.labelled_image_input.get_sized_training_batch_set(num_batches, batch_size, (224,224))
        for image_labels, img_lbl_vectors, flag_vectors, img_224,  in batched_image_input:
            disc_loss = self.__run_batch__(img_224, image_labels, img_lbl_vectors, training=False)
            disc_loss_bucket.append(disc_loss)
        self.log_testing_tensorboard_data()
        return np.sum(np.mean(disc_loss_bucket,axis=0))
    
    def save_progressive_real_images(self, filename):
        self.__save_progressive_classified_images(filename, self.most_recent_output)

    def __save_progressive_classified_images(self, filename, prog_images_with_labels_and_preds):
        ##currently only implemented for single images
        num_plots = len(prog_images_with_labels_and_preds)
        num_imgs = len(prog_images_with_labels_and_preds[0][0])
        fig,axes = plt.subplots(num_imgs, num_plots, figsize=(num_plots*16, num_imgs*16))
        i=0
        for image_set, given_labels, predicted_labels in prog_images_with_labels_and_preds:
            for j, img in enumerate(image_set):
                predicted_lbl = self.labelled_image_input.label_vectorizer.get_label_names_by_ids(predicted_labels[j])
                pass_fail = "PASS" if predicted_lbl == given_labels[j] else "FAIL"
                text_label = predicted_lbl + " || " + given_labels[j] + " || " + pass_fail
                img = np.array(img)
                img = scale_to_255(img)
                img = Image.fromarray((img).astype(np.uint8))
                img = np.asarray(img)
                if num_imgs > 1 and num_plots > 1:
                    axes[j,i].imshow(img)
                    axes[j,i].set_title(text_label, fontsize=32)
                elif num_plots > 1:
                    axes[i].imshow(img)
                    axes[i].set_title(text_label, fontsize=32)
                else:
                    axes[j].imshow(img)
                    axes[j].set_title(text_label, fontsize=32)
            i+=1
        fig.savefig(filename)
        plt.close()
    
    def log_training_tensorboard_data(self, disc_loss):
        with self.writer.as_default():
            tf.summary.scalar("disc_loss", disc_loss, step=self.step)
            
            disc_WAs = [x for x in get_all_layers(self.discriminator) if 'weighted_add' in x.name]
            # print(disc_WAs)
            for disc_WA in disc_WAs:
                if hasattr(disc_WA, "input_shape") and disc_WA.input_shape is not None:
                    input_shape = disc_WA.input_shape[0]
                else:
                    input_shape = disc_WA._serialized_attributes["metadata"]["build_input_shape"][0]
                lbl = f"disc_weighted_add/{'x'.join([str(x) for x in input_shape[1:-1]])}"
                tf.summary.histogram(lbl, disc_WA.a.numpy(), step=self.step)
        self.step += 1
    
    def log_testing_tensorboard_data(self):
        with self.writer.as_default():
            for disc_real_img, _, _ in self.most_recent_output:
                lbl = f"disc_output_{'x'.join([str(x) for x in disc_real_img.shape[1:3]])}"
                tf.summary.image(lbl, disc_real_img, self.step)
        self.step += 1