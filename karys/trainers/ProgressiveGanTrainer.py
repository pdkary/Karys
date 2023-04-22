import numpy as np
from typing import List
import tensorflow as tf
from keras.models import Model
from keras.optimizers import Optimizer
from keras.losses import Loss 
import keras.backend as K
from PIL import Image
import matplotlib.pyplot as plt

from karys.data.ImageDataLoader import ImageDataLoader
from karys.layers.LearnedNoise import LearnedNoise
from karys.layers.WeightedAdd import WeightedAdd

def scale_to_255(img):
    img_min = np.min(img)
    img_max = np.max(img)
    return 255*(img - img_min)/(img_max - img_min + 1e-8)

@tf.RegisterGradient("ResizeNearestNeighborGrad")
def _ResizeNearestNeighborGrad_grad(op, grad):
    up = tf.image.resize(grad,tf.shape(op.inputs[0])[1:-1])
    return up,None

def get_all_layers(model: Model):
    out_layers = []
    for x in model.layers:
        if hasattr(x, "layers"):
            out_layers.extend(get_all_layers(x))
        else:
            out_layers.append(x)
    return out_layers

class ProgressiveGanTrainer(object):
    def __init__(self,
                 generator: Model,
                 discriminator: Model,
                 gen_optimizer: Optimizer,
                 gen_loss: List[Loss],
                 disc_optimizer: Optimizer,
                 disc_loss: List[Loss],
                 style_loss: List[Loss],
                 labelled_image_input: ImageDataLoader,
                 tensorboard_writer):
        self.generator = generator
        self.discriminator = discriminator
        self.labelled_image_input = labelled_image_input
        self.image_shape = discriminator.input_shape[1:-1]

        self.gen_optimizer: Optimizer = gen_optimizer
        self.gen_loss: Loss = gen_loss
        self.disc_optimizer: Optimizer = disc_optimizer
        self.disc_loss: Loss = disc_loss
        self.style_loss: Loss = style_loss

        self.writer = tensorboard_writer
        self.step = 0
        
        print("Generator input shape: ", generator.input_shape)
        print("Generator output shape: ", generator.output_shape)
        print("Discriminator input shape: ", discriminator.input_shape)
        print("Discriminator output shape: ", discriminator.output_shape)
        self.generator.compile(optimizer=gen_optimizer, loss=[gen_loss for d in generator.output_shape])
        self.discriminator.compile(optimizer=disc_optimizer, loss=[disc_loss for d in discriminator.output_shape])

        self.most_recent_gen_output = None
        self.most_recent_gen_input = None
    
    def __get_training_outputs__(self, noise_batch, img_lbls_vecs,  img_224, disc_training=False, gen_training=False):
        generator_out = self.generator([noise_batch,img_lbls_vecs], training=gen_training)
        G, G_hidden = generator_out[0], generator_out[1:]

        disc_gen_out = self.discriminator(G, training=disc_training)
        disc_real_out = self.discriminator(img_224, training=disc_training)

        Dg, Dr = disc_gen_out[0], disc_real_out[0]
        Fg, Fr = disc_gen_out[-1], disc_real_out[-1]

        Dg_hidden, Dr_hidden = disc_gen_out[1:-1], disc_real_out[1:-1]
        return G, Dg, Dr, G_hidden, Dr_hidden, Dg_hidden, Fg, Fr
    
    def __run_batch__(self, batched_noise_input, batched_image_input, disc_training=True, gen_training=True):
        m = 0
        alpha = 0.25
        gen_loss, style_loss, disc_loss = 0, 0, 0
        for noise_batch, (image_labels, img_lbl_vectors, img_224) in zip(batched_noise_input, batched_image_input):
            G, Dg, Dr, G_hidden, Dr_hidden, Dg_hidden, Fg, Fr = self.__get_training_outputs__(noise_batch, img_lbl_vectors, img_224, disc_training=disc_training, gen_training=gen_training)
            disc_loss += self.disc_loss(0.9*img_lbl_vectors, Dr) + self.disc_loss(tf.zeros_like(img_lbl_vectors), Dg)
            disc_loss /= len(noise_batch)
            style_loss += self.get_style_loss(Dr_hidden, Dg_hidden) / len(noise_batch)
            style_loss += self.gen_loss(K.std(Fr, axis=0), K.std(Fg, axis=0)) / len(noise_batch)
            style_loss += self.gen_loss(K.mean(Fr, axis=0), K.mean(Fg, axis=0)) / len(noise_batch)
            gen_loss += self.gen_loss(alpha*tf.ones_like(Dg)+(1-alpha)*img_lbl_vectors, Dg) / len(noise_batch)
            m += 1
        
        gen_loss /= m
        disc_loss /= m
        style_loss /= m
        argmax_Dg, argmax_Dr = K.argmax(Dg).numpy(), K.argmax(Dr).numpy()
        
        flag_labels = ["generated" for x in image_labels]
        self.most_recent_real_disc_output = []
        self.most_recent_gen_disc_output = []
        self.most_recent_gen_output = [(G, flag_labels, argmax_Dg)]
        
        for i in range(len(Dr_hidden)):
            self.most_recent_real_disc_output.append(( Dr_hidden[i], image_labels, argmax_Dr ))
        for i in range(len(Dg_hidden)):
            self.most_recent_gen_disc_output.append(( Dg_hidden[i], image_labels, argmax_Dg ))
        for i in range(len(G_hidden)):
            self.most_recent_gen_output.append(( G_hidden[i], flag_labels, argmax_Dg ))

        if gen_training or disc_training:
            self.log_training_tensorboard_data(gen_loss, style_loss, disc_loss, image_labels, Fr, Fg)
        return gen_loss, style_loss, disc_loss

    
    def __run_multi_batch__(self, batched_noise_input, batched_image_input, disc_training = True, gen_training = True):
        ##actual training
        with tf.GradientTape() as genr_tape, tf.GradientTape() as disc_tape:
            gen_loss, style_loss, disc_loss = self.__run_batch__(batched_noise_input, batched_image_input, disc_training, gen_training)            
            gen_grads = genr_tape.gradient(gen_loss + style_loss, self.generator.trainable_variables)
            disc_grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.gen_optimizer.apply_gradients(zip(gen_grads, self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(disc_grads, self.discriminator.trainable_variables))
        
        return gen_loss, style_loss, disc_loss

    
    def train(self, batch_size, num_batches) -> tf.float32:
        batched_image_input = self.labelled_image_input.get_sized_training_batch_set(num_batches, batch_size, self.discriminator.input_shape[1:-1])
        batched_noise_input = np.random.normal(0.0, 1.0,size=(num_batches, batch_size, self.generator.input_shape[0][-1]))
        gen_loss, style_loss, disc_loss = self.__run_multi_batch__(batched_noise_input, batched_image_input, True, True)
        return gen_loss, style_loss, disc_loss

    def test(self, batch_size, num_batches):
        batched_image_input = self.labelled_image_input.get_sized_validation_batch_set(num_batches, batch_size, self.discriminator.input_shape[1:-1])
        batched_noise_input = np.random.normal(0.0,1.0,size=(num_batches, batch_size, self.generator.input_shape[0][-1]))
        gen_loss, style_loss, disc_loss = self.__run_batch__(batched_noise_input, batched_image_input, False, False)
        self.log_testing_tensorboard_data()
        return gen_loss, style_loss, disc_loss
    
    def get_style_loss(self, real_hidden, gen_hidden):
        style_loss = 0
        for rh, gh in zip(real_hidden, gen_hidden):
            style_loss += self.gen_loss(K.std(rh, axis=0), K.std(gh, axis=0))
            style_loss += self.gen_loss(K.mean(rh, axis=0), K.mean(gh, axis=0))
        style_loss /= len(real_hidden)
        return style_loss

    def save_progressive_disc_gen_images(self, filename):
        self.__save_progressive_classified_images(filename, self.most_recent_gen_disc_output)

    def save_progressive_gen_images(self, filename):
        self.__save_progressive_classified_images(filename, self.most_recent_gen_output)
    
    def save_progressive_disc_real_images(self, filename):
        self.__save_progressive_classified_images(filename, self.most_recent_real_disc_output)

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

                if axes.ndim > 1:
                    axes[j,i].imshow(img)
                    axes[j,i].set_title(text_label, fontsize=32)
                else:
                    axes[j].imshow(img)
                    axes[j].set_title(text_label, fontsize=32)
            i+=1
        fig.savefig(filename)
        plt.close()
    
    def log_training_tensorboard_data(self, gen_loss, style_loss, disc_loss, image_labels, real_features, generated_features):
        with self.writer.as_default():
            tf.summary.scalar("gen_loss", gen_loss, step=self.step)
            tf.summary.scalar("style_loss", style_loss, step=self.step)
            tf.summary.scalar("disc_loss", disc_loss, step=self.step)
            
            for lbl, real_feature, gen_feature in zip(image_labels, real_features, generated_features):
                # print(real_feature)
                tf.summary.histogram(f"generated_label_features/{lbl}", real_feature.numpy(), step=self.step)
                tf.summary.histogram(f"real_label_features/{lbl}", gen_feature.numpy(), step=self.step)
            
            disc_WAs = [x for x in get_all_layers(self.discriminator) if 'weighted_add' in x.name]
            # print(disc_WAs)
            for disc_WA in disc_WAs:
                if hasattr(disc_WA, "input_shape") and disc_WA.input_shape is not None:
                    input_shape = disc_WA.input_shape[0]
                else:
                    input_shape = disc_WA._serialized_attributes["metadata"]["build_input_shape"][0]
                lbl = f"disc_weighted_add/{'x'.join([str(x) for x in input_shape[1:-1]])}"
                tf.summary.histogram(lbl, disc_WA.a.numpy(), step=self.step)

            gen_WAs = [x for x in get_all_layers(self.generator) if 'weighted_add' in x.name]
            # print(gen_WAs)
            for gen_WA in gen_WAs:
                if hasattr(gen_WA, "input_shape") and gen_WA.input_shape is not None:
                    if issubclass(type(gen_WA), LearnedNoise):
                        input_shape = gen_WA.input_shape
                    elif issubclass(type(gen_WA), WeightedAdd):
                        input_shape = gen_WA.input_shape[0]
                else:
                    model_metadata = gen_WA._serialized_attributes["metadata"]
                    if model_metadata['class_name'] == "LearnedNoise":
                        input_shape = model_metadata["build_input_shape"]
                    elif model_metadata['class_name'] == "WeightedAdd":
                        if hasattr(model_metadata,"build_input_shape"):
                            input_shape = model_metadata["build_input_shape"][0]
                        else:
                            continue
                
                lbl = f"gen_weighted_add/{'x'.join([str(x) for x in input_shape[1:-1]])}"
                tf.summary.histogram(lbl, gen_WA.a.numpy(), step=self.step)

            gen_ADAs = [x for x in get_all_layers(self.generator) if 'adaptive' in x.name]

            for gen_ADA in gen_ADAs:
                if hasattr(gen_ADA, "input_shape") and gen_ADA.input_shape is not None:
                    input_shape = gen_ADA.input_shape
                else:
                    input_shape = gen_ADA._serialized_attributes["metadata"]["build_input_shape"]
                
                b_lbl = f"gen_ADA_beta/{'x'.join([str(x) for x in input_shape[1:]])}"
                g_lbl = f"gen_ADA_gamma/{'x'.join([str(x) for x in input_shape[1:]])}"
                tf.summary.histogram(b_lbl, gen_ADA.beta.numpy(), step=self.step)
                tf.summary.histogram(g_lbl, gen_ADA.gamma.numpy(), step=self.step)
            
            
            disc_ADAs = [x for x in get_all_layers(self.discriminator) if 'adaptive' in x.name]

            for disc_ADA in disc_ADAs:
                if hasattr(disc_ADA, "input_shape") and disc_ADA.input_shape is not None:
                    input_shape = disc_ADA.input_shape
                else:
                    input_shape = disc_ADA._serialized_attributes["metadata"]["build_input_shape"]
                
                b_lbl = f"disc_ADA_beta/{'x'.join([str(x) for x in input_shape[1:]])}"
                g_lbl = f"disc_ADA_gamma/{'x'.join([str(x) for x in input_shape[1:]])}"
                tf.summary.histogram(b_lbl, disc_ADA.beta.numpy(), step=self.step)
                tf.summary.histogram(g_lbl, disc_ADA.gamma.numpy(), step=self.step)
        self.step += 1
        
    def log_testing_tensorboard_data(self):
        with self.writer.as_default():
            for disc_real_img, _, _ in self.most_recent_real_disc_output:
                lbl = f"disc_real_output_{'x'.join([str(x) for x in disc_real_img.shape[1:3]])}"
                tf.summary.image(lbl, disc_real_img, self.step)

            for disc_gen_img, _, _ in self.most_recent_gen_disc_output:
                lbl = f"disc_gen_output_{'x'.join([str(x) for x in disc_gen_img.shape[1:3]])}"
                tf.summary.image(lbl, disc_gen_img, self.step)
            
            for gen_img, _, _ in self.most_recent_gen_output:
                if type(gen_img) == list:
                    gen_img = gen_img[0]
                lbl = f"gen_output_{'x'.join([str(x) for x in gen_img.shape[1:3]])}"
                tf.summary.image(lbl, gen_img, self.step)
        self.step += 1
