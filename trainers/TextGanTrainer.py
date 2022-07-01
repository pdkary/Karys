import tensorflow as tf

from trainers.abstract_trainers.AbstractGanTrainer import AbstractGanTrainer

class TextGanTrainer(AbstractGanTrainer):
    ##in this case, we are using the same input for the generator and the discriminator
    ##the text data wrapper will give us ((None,NI,I),(None,NO,O)) of input sentences and their subsequent (output) sentences
    def test_step(self,gen_input, gen_labels, disc_input, disc_labels):
        #gen input is out (None,NI,I)
        gen_out = self.generator.model(gen_input, training=True)
        #we've now generated a new (None,NO,O')
        self.most_recent_gen_output = gen_out
        self.most_recent_gen_input = gen_input
        ##discriminate the real (None,NO,O)
        discriminated_real_images = self.discriminator.model(gen_labels, training=True)
        ##discriminate the generated (None,NO,O')
        discriminated_genr_images = self.discriminator.model(gen_out, training=True)

        ##ideally, the generator would produce images that would get discriminated as all 1s
        genr_loss = self.generator.loss(tf.ones_like(discriminated_genr_images), discriminated_genr_images)
        ##ideally, the discriminator would identify real images as all 1s
        disc_real_loss = self.discriminator.loss(tf.ones_like(discriminated_real_images), discriminated_real_images)
        ##ideally, the discriminator would identify generated images as all 0s
        disc_genr_loss = self.discriminator.loss(tf.zeros_like(discriminated_genr_images),discriminated_genr_images)
        disc_loss = disc_genr_loss + disc_real_loss
        return genr_loss, disc_loss



        

