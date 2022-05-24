from data.wrappers.RandomDataWrapper import RandomDataWrapper
from models.GanModelBase import GanModelBase
import tensorflow as tf

class GanGeneratorModel(GanModelBase):
    def __post__init__(self):
        self.most_recent_gen_test = None

    def get_noise_dataset(self) -> tf.data.Dataset:
        if self.datawrapper is None:
            raise ValueError(
                "datawrapper and noise_size must be set before getting noise dataset")
        else:
            return self.datawrapper.get_dataset().batch(self.dataconfig.batch_size)

    def get_train_dataset(self) -> tf.data.Dataset:
        return self.get_noise_dataset().shuffle(buffer_size=512).take(self.dataconfig.num_batches)
    
    def get_test_dataset(self) -> tf.data.Dataset:
        return self.get_noise_dataset().take(self.dataconfig.num_batches)
        
    def set_discriminator(self, discriminator):
        self.discriminator = discriminator
    
    def evaluate_train_loss(self, model_input, labels, model_output):
        ##labels are always 1s
        gen_images = model_output
        disc_output = self.discriminator.model(gen_images,training=False)
        loss = self.loss(tf.ones_like(disc_output),disc_output)
        return loss

    def evaluate_test_loss(self, model_input, labels, model_output):
        self.most_recent_gen_test = model_output
        gen_images = model_output
        disc_output = self.discriminator.model(gen_images,training=False)
        loss = self.loss(tf.ones_like(disc_output),disc_output)
        return loss
    
