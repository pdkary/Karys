from time import time

import numpy as np
from data.configs.ImageDataConfig import ImageDataConfig
from data.wrappers.ImageDataWrapper import ImageDataWrapper
from keras.losses import BinaryCrossentropy, MeanSquaredError
from keras.models import load_model
from keras.optimizers import Adam
from data.configs.RandomDataConfig import RandomDataConfig
from data.wrappers.RandomDataWrapper import RandomDataWrapper
from trainers.ImageGanTrainer import ImageGanTrainer
from models.vgg16 import Vgg16Classifier, ReverseVgg16Generator

base_path = "./examples/generative_adversarial"
input_path = base_path + "/test_input/"
output_path = "./examples/generative_adversarial/test_output"

gen_path = output_path + "/vgg16_bob_generator"
disc_path = output_path + "/vgg16_bob_discriminator"

def load_data():
    extra_labels = ["generated"]
    image_config = ImageDataConfig(image_shape=(224,224, 3),image_type=".png", preview_rows=3, preview_cols=3, load_n_percent=100)
    data_wrapper = ImageDataWrapper.load_from_labelled_directories(input_path + '/', image_config, extra_labels, validation_percentage=0.1)

    random_config = RandomDataConfig([512], 0.0, 1.5)
    random_data_wrapper = RandomDataWrapper(random_config)
    return data_wrapper, random_data_wrapper

def load_models(num_categories, noise_size):
    try:
        print("loading classifier...")
        classifier = load_model(disc_path)
    except Exception as e:
        classifier = Vgg16Classifier(num_categories).build_graph()
    classifier.summary()
    try:
        print("loading generator...")
        generator = load_model(gen_path)
    except:
        print("loading of generator failed, building...")
        generator = ReverseVgg16Generator(noise_size).build_graph()
    generator.summary()
    return generator, classifier

def train(epochs, trains_per_test):
    data_wrapper, random_data_wrapper = load_data()
    generator, classifier = load_models(data_wrapper.label_dim, random_data_wrapper.data_config.shape[0])
    
    gen_optimizer = Adam(learning_rate=2e-5)
    disc_optimizer = Adam(learning_rate=2e-5)

    style_loss_fn = MeanSquaredError(reduction="sum")
    gen_loss_fn = BinaryCrossentropy(from_logits=True, reduction="sum")
    disc_loss_fn = BinaryCrossentropy(from_logits=True, reduction="sum")
    trainer = ImageGanTrainer(generator, classifier, gen_optimizer, gen_loss_fn, disc_optimizer, disc_loss_fn, style_loss_fn, random_data_wrapper, data_wrapper)
    for i in range(epochs):
        start_time = time()
        gen_loss, disc_loss, gen_style_loss, disc_style_loss = trainer.train(4, 2)
        avg_gen_loss = np.mean(gen_loss)
        avg_disc_loss = np.mean(disc_loss)
        avg_gen_style_loss = np.mean(gen_style_loss)
        avg_disc_style_loss = np.mean(disc_style_loss)

        if i % trains_per_test == 0 and i != 0:
            test_gen_loss, test_disc_loss, gen_style_loss, disc_style_loss = trainer.test(9,1)
            avg_gen_loss = np.mean(test_gen_loss)
            avg_disc_loss = np.mean(test_disc_loss)
            avg_gen_style_loss = np.mean(gen_style_loss)
            avg_disc_style_loss = np.mean(disc_style_loss)
            real_output_filename = output_path + "/real-" + str(i) + ".jpg"
            gen_output_filename = output_path + "/generated-" + str(i) + ".jpg"
            data_wrapper.save_classified_images(real_output_filename, trainer.most_recent_output, img_size=32)
            data_wrapper.save_classified_images(gen_output_filename, trainer.most_recent_gen_output, img_size=128)
        
        end_time = time()
        test_label = "TEST" if i % trains_per_test == 0 and i != 0 else ""
        print(f"EPOCH {i}/{epochs}: gen loss={avg_gen_loss}, gen style_loss={avg_gen_style_loss}, disc style_loss={avg_disc_style_loss}, disc loss={avg_disc_loss}, time={end_time-start_time}, {test_label}")
    classifier.save(disc_path)
    generator.save(gen_path)