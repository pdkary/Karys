import numpy as np
import tensorflow as tf
from data.labels.CategoricalLabel import CategoricalLabel


from data.wrappers.ImageDataWrapper import ImageDataWrapper
from data.wrappers.RandomDataWrapper import RandomDataWrapper
from models.ClassificationModel import ClassificationModel
from models.GenerationModel import GenerationModel

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

class ImageGanTrainer(object):
    def __init__(self,
                 generator: GenerationModel,
                 discriminator: ClassificationModel,
                 noise_input: RandomDataWrapper,
                 labelled_image_input: ImageDataWrapper):
        self.generator = generator
        self.discriminator = discriminator
        self.noise_input = noise_input
        self.labelled_image_input = labelled_image_input

        self.noise_train_dataset = self.noise_input.get_train_dataset()
        self.noise_test_dataset = self.noise_input.get_validation_dataset()
        self.image_train_dataset = self.labelled_image_input.get_train_dataset()
        self.image_test_dataset = self.labelled_image_input.get_validation_dataset()
        self.most_recent_gen_output = None
        self.most_recent_gen_input = None

    def __run_batch__(self, noise_batch, image_batch, image_names, training=False):
        gen_out = self.generator.generate(noise_batch, training=training)
        self.most_recent_gen_output = gen_out        
        
        label_func = lambda x : self.discriminator.label_generator.get_single_category(self.discriminator.label_to_id[x])
        batch_labels = [ self.labelled_image_input.image_labels[n] for n in image_names]
       
        real_labels = np.array([label_func(l) for l in batch_labels])
        fake_labels = np.array([label_func("FAKE") for l in batch_labels])

        _, gen_probs, gen_preds = self.discriminator.classify(gen_out, training)
        _, real_probs, _ = self.discriminator.classify(image_batch, training)

        genr_loss = self.generator.loss(real_labels, gen_probs)
        disc_real_loss = self.discriminator.loss(real_labels, real_probs)
        disc_genr_loss = self.discriminator.loss(fake_labels, gen_probs)

        disc_loss = disc_genr_loss + disc_real_loss

        self.most_recent_output = list(zip(gen_out, gen_preds))
        return genr_loss, disc_loss

    def train(self, batch_size, num_batches) -> np.float32:
        gen_loss_bucket = []
        disc_loss_bucket = []

        batched_noise_input = batch(self.noise_train_dataset, batch_size)
        batched_image_input = batch_dict(self.image_train_dataset, batch_size)
        np.random.shuffle(batched_noise_input)
        np.random.shuffle(batched_image_input)

        noise_input_dataset = batched_noise_input[:num_batches]
        image_input_dataset = batched_image_input[:num_batches]

        for noise_batch, (image_names, image_batch) in zip(noise_input_dataset, image_input_dataset):
            with tf.GradientTape() as genr_tape, tf.GradientTape() as disc_tape:
                gen_loss, disc_loss = self.__run_batch__(noise_batch, image_batch, image_names)
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

        batched_noise_input = batch(self.noise_train_dataset, batch_size)
        batched_image_input = batch_dict(self.image_train_dataset, batch_size)
        np.random.shuffle(batched_noise_input)
        np.random.shuffle(batched_image_input)

        noise_input_dataset = batched_noise_input[:num_batches]
        image_input_dataset = batched_image_input[:num_batches]

        for noise_batch, (image_names, image_batch) in zip(noise_input_dataset, image_input_dataset):
            gen_loss, disc_loss = self.__run_batch__(noise_batch, image_batch, image_names)
            gen_loss_bucket.append(gen_loss)
            disc_loss_bucket.append(disc_loss)
        return np.sum(np.mean(gen_loss_bucket,axis=0)), np.sum(np.mean(disc_loss_bucket,axis=0))
        



        

