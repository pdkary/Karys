

from time import time

import dictdiffer
import numpy as np
from karys.data.wrappers.RandomDataWrapper import RandomDataWrapper
from karys.data.configs.ImageDataConfig import ImageDataConfig
from karys.data.configs.RandomDataConfig import RandomDataConfig
from karys.data.wrappers.ImageDataWrapper import ImageDataWrapper
from karys.models.vgg16 import ReverseVgg16Generator, Vgg16Classifier
from keras.losses import BinaryCrossentropy
from keras.models import load_model
from keras.optimizers import Adam
from trainers.ImageGanTrainer import ImageGanTrainer


def test_train_gan_model(generator, discriminator, gen_optimizer, gen_loss, disc_optimizer, disc_loss, noise_data_wrapper, image_data_wrapper, epochs=5, trains_per_test=4):
    trainer = ImageGanTrainer(generator, discriminator, gen_optimizer, gen_loss, disc_optimizer, disc_loss, noise_data_wrapper, image_data_wrapper)
    output_path = "./examples/discriminator/test_output"

    for i in range(epochs):
        start_time = time()
        gen_loss, disc_loss = trainer.train(4, 8)
        avg_gen_loss = np.mean(gen_loss)
        avg_disc_loss = np.mean(disc_loss)

        if i % trains_per_test == 0 and i != 0:
            test_gen_loss, test_disc_loss = trainer.test(16,1)
            avg_gen_loss = np.mean(test_gen_loss)
            avg_disc_loss = np.mean(test_disc_loss)
            real_output_filename = output_path + "/real-" + str(i) + ".jpg"
            gen_output_filename = output_path + "/generated-" + str(i) + ".jpg"
            image_data_wrapper.save_classified_images(real_output_filename, trainer.most_recent_output, img_size=32)
            image_data_wrapper.save_classified_images(gen_output_filename, trainer.most_recent_gen_output, img_size=32)
        
        end_time = time()
        test_label = "TEST" if i % trains_per_test == 0 and i != 0 else ""
        print(f"EPOCH {i}/{epochs}: {test_label} gen loss={avg_gen_loss}, disc loss={avg_disc_loss}, time={end_time-start_time}")

def test_gan_model():
    ##set up data
    base_path = "./examples/discriminator/test_input/Fruit"
    extra_labels = ["generated"]
    image_config = ImageDataConfig(image_shape=(224,224, 3),image_type=".jpg", preview_rows=4, preview_cols=4, load_n_percent=20)
    image_data_wrapper = ImageDataWrapper.load_from_labelled_directories(base_path + '/', image_config, extra_labels, validation_percentage=0.1)

    random_config = RandomDataConfig([1024], 0.0, 1.0)
    random_data_wrapper = RandomDataWrapper(random_config)
    
    ## Build Model
    generator = ReverseVgg16Generator(random_config.shape[0]).build_graph()
    generator.summary()

    discriminator = Vgg16Classifier(len(image_data_wrapper.classification_labels)).build_graph()
    discriminator.summary()

    gen_optimizer = Adam(learning_rate=4e-5)
    gen_loss = BinaryCrossentropy(from_logits=True,reduction="sum_over_batch_size")
    disc_optimizer = Adam(learning_rate=4e-5)
    disc_loss = BinaryCrossentropy(from_logits=True,reduction="sum_over_batch_size")

    ## Train model
    epochs=10
    trains_per_test=4
    test_train_gan_model(generator, discriminator, gen_optimizer, gen_loss, disc_optimizer,disc_loss, random_data_wrapper, image_data_wrapper, epochs, trains_per_test)

    ## Save model
    output_path = "./examples/discriminator/test_output"
    generator.save(output_path + "/vgg16_fruit_generator")
    discriminator.save(output_path + "/vgg16_fruit_discriminator")

    # Reload model
    new_generator = load_model(output_path + "/vgg16_fruit_generator")
    new_discriminator = load_model(output_path + "/vgg16_fruit_discriminator")

    ## Train model again
    test_train_gan_model(new_generator, new_discriminator, gen_optimizer, gen_loss, disc_optimizer,disc_loss, random_data_wrapper, image_data_wrapper, epochs, trains_per_test)
    
    ## Save model again
    generator.save(output_path + "/vgg16_fruit_generator")
    discriminator.save(output_path + "/vgg16_fruit_discriminator")
