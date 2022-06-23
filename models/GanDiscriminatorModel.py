from models.bases.GanModelBase import GanModelBase
import tensorflow as tf

class GanDiscriminatorModel(GanModelBase):
    def get_train_dataset(self) -> tf.data.Dataset:
        if self.datawrapper is None:
            raise ValueError(
                "datawrapper and dataconfig must be set before getting train dataset")
        else:
            return self.datawrapper.get_train_dataset()
    
    def get_test_dataset(self) -> tf.data.Dataset:
        if self.datawrapper is None:
            raise ValueError(
                "datawrapper and dataconfig must be set before getting test dataset")
        else:
            return self.datawrapper.get_test_dataset()

    def set_generator(self, generator):
        self.generator = generator
    
    def evaluate_train_loss(self, model_input, labels, model_output):
        gen_noise_batch = self.generator.get_datawrapper().get_single()
        
        disc_real_output = model_output
        gen_output = self.generator.model(gen_noise_batch, training=False)
        disc_gen_output = self.model(gen_output, training=True)
        
        real_loss = self.loss(tf.ones_like(disc_real_output), disc_real_output)
        fake_loss = self.loss(tf.zeros_like(disc_gen_output), disc_gen_output)

        loss = real_loss + fake_loss
        return loss

    def evaluate_test_loss(self, model_input, labels, model_output):
        ##labels are always 1s
        gen_noise_batch = self.generator.get_datawrapper().get_single()
        disc_real_output = model_output
        gen_output = self.generator.model(gen_noise_batch, training=False)
        disc_gen_output = self.model(gen_output, training=True)

        real_loss = self.loss(tf.ones_like(disc_real_output), disc_real_output)
        fake_loss = self.loss(tf.zeros_like(disc_gen_output), disc_gen_output)

        loss = real_loss + fake_loss
        return loss