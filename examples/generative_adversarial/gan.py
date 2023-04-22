from time import time
from typing import Tuple

import numpy as np
from karys.data.ImageDataLoader import ImageDataLoader
from karys.data.configs.ImageDataConfig import ImageDataConfig
from karys.data.wrappers.ImageDataWrapper import ImageDataWrapper
from keras.losses import BinaryCrossentropy, MeanSquaredError
from keras.models import load_model, Model
from keras.optimizers import Adam
from karys.data.configs.RandomDataConfig import RandomDataConfig
from karys.data.wrappers.RandomDataWrapper import RandomDataWrapper
from karys.trainers.ImageGanTrainer import ImageGanTrainer
from karys.models.vgg16 import Vgg16Classifier, ReverseVgg16Generator

base_path = "./examples/discriminator"
input_path = base_path + "/test_input/Fruit"
output_path = "./examples/generative_adversarial/test_output"

gen_path = output_path + "/vgg16_bob_generator"
disc_path = output_path + "/vgg16_bob_discriminator"

noise_size = 512

gen_optimizer = Adam(learning_rate=2e-5)
disc_optimizer = Adam(learning_rate=2e-5)

style_loss_fn = MeanSquaredError(reduction="sum")
gen_loss_fn = BinaryCrossentropy(from_logits=True, reduction="sum")
disc_loss_fn = BinaryCrossentropy(from_logits=True, reduction="sum")

def load_models() -> Tuple[Model, Model, ImageGanTrainer]:
    global noise_size, gen_path, disc_path
    data_loader = ImageDataLoader(input_path + "/",".jpg",0.1)
    classification_size = len(data_loader.label_set)
    try:
        print("loading classifier...")
        classifier = load_model(disc_path)
    except Exception as e:
        classifier = Vgg16Classifier(classification_size).build_graph()
    classifier.summary()
    try:
        print("loading generator...")
        generator = load_model(gen_path)
    except:
        print("loading of generator failed, building...")
        generator = ReverseVgg16Generator(noise_size).build_graph()
    generator.summary()
    trainer = ImageGanTrainer(generator, classifier, gen_optimizer, gen_loss_fn, disc_optimizer, disc_loss_fn, style_loss_fn,  data_loader)
    return generator, classifier, trainer

def train(epochs, trains_per_test):
    global gen_loss_fn, disc_optimizer, disc_loss_fn, style_loss_fn
    generator, classifier, trainer = load_models()
    
    for i in range(epochs):
        start_time = time()
        gen_loss, disc_loss, gen_style_loss, disc_style_loss = trainer.train(6, 1)
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
            trainer.save_classified_real_images(real_output_filename, 3,3, img_size=128)
            trainer.save_classified_gen_images(gen_output_filename, 3,3, img_size=128)
        
        end_time = time()
        test_label = "TEST" if i % trains_per_test == 0 and i != 0 else ""
        print(f"EPOCH {i}/{epochs}: gen loss={avg_gen_loss}, gen style_loss={avg_gen_style_loss}, disc style_loss={avg_disc_style_loss}, disc loss={avg_disc_loss}, time={end_time-start_time}, {test_label}")
    classifier.save(disc_path)
    generator.save(gen_path)
