import numpy as np
import tensorflow as tf

from trainers.abstract_trainers.AbstractGanTrainer import AbstractGanTrainer

class ImageGanTrainer(AbstractGanTrainer):
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



        

