from time import time

import numpy as np
from data.configs.ImageDataConfig import ImageDataConfig
from data.wrappers.ImageDataWrapper import ImageDataWrapper
from keras.layers import (Activation, Conv2D, Dense, Flatten, LeakyReLU, MaxPooling2D)
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
from models.transformer import ClassificationModel
from models.blocking import Conv2DBatchNormLeakyReluBlock
from trainers.ClassificationTrainer import ClassificationTrainer


def train(data_wrapper, classifier, epochs, trains_per_test):
    trainer = ClassificationTrainer(classifier, data_wrapper)
    image_output_path = "./examples/discriminator/test_output"
    test_loss = 0
    for i in range(epochs):
        start_time = time()
        loss = trainer.train(32, 3)
        avg_loss = np.mean(loss)

        if i % trains_per_test == 0 and i != 0:
            test_loss = trainer.test(16,1)
            avg_loss = np.mean(test_loss)
            test_output_filename = image_output_path + "/train-" + str(i) + ".jpg"
            data_wrapper.save_classified_images(test_output_filename, trainer.most_recent_output, img_size=32)
        
        end_time = time()
        print(f"EPOCH {i}/{epochs}: loss={avg_loss}, time={end_time-start_time}")
    classifier.save(image_output_path)

def build_classifier(base_path, image_type=".jpg"):
    image_config = ImageDataConfig(image_shape=(224,224, 3),image_type=image_type, preview_rows=4, preview_cols=4, load_n_percent=100)
    data_wrapper = ImageDataWrapper.load_from_labelled_directories(base_path + '/', image_config, validation_percentage=0.1)
    classification_labels = list(set(data_wrapper.image_labels.values()))
    a = 0.08
    disc_layers = [
        Conv2DBatchNormLeakyReluBlock(2, a, dict(filters=64,kernel_size=3, padding="same")),
        MaxPooling2D(),
        Conv2DBatchNormLeakyReluBlock(2, a, dict(filters=128,kernel_size=3, padding="same")),
        MaxPooling2D(),
        Conv2DBatchNormLeakyReluBlock(3, a, dict(filters=256,kernel_size=3, padding="same")),
        MaxPooling2D(),
        Conv2DBatchNormLeakyReluBlock(3, a, dict(filters=512,kernel_size=3, padding="same")),
        MaxPooling2D(),
        Conv2DBatchNormLeakyReluBlock(3, a, dict(filters=512,kernel_size=3, padding="same")),
        MaxPooling2D(),
        Flatten(),
        Dense(4096), Activation('relu'),
        Dense(4096), Activation('relu'),
        Dense(len(classification_labels)+1), Activation('softmax'),
    ]
    optimizer = Adam(learning_rate=4e-5)
    loss = BinaryCrossentropy(from_logits=True,reduction="sum_over_batch_size")
    classifier = ClassificationModel(image_config.image_shape, classification_labels, ['generated'], disc_layers, optimizer, loss, name="fruit_classifier")
    classifier.build()
    return data_wrapper, classifier

def build_and_train(epochs, trains_per_test):
    data_wrapper, classifier = build_classifier()
    train(data_wrapper, classifier, epochs, trains_per_test)

def load_and_train(epochs, trains_per_test):
    base_path = "./examples/discriminator"
    model_path = base_path + "/test_output/fruit_classifier"
    image_path = base_path + "/test_input/Fruit"
    optimizer = Adam(learning_rate=4e-5)
    loss = BinaryCrossentropy(from_logits=True,reduction="sum_over_batch_size")
    classifier = ClassificationModel.load_from_filepath(model_path, optimizer, loss)
    classifier.build()

    image_config = ImageDataConfig(image_shape=(224,224, 3),image_type=".jpg", preview_rows=4, preview_cols=4, load_n_percent=100)
    data_wrapper = ImageDataWrapper.load_from_labelled_directories(image_path + '/', image_config, validation_percentage=0.1)
    train(data_wrapper, classifier, epochs, trains_per_test)




    
        