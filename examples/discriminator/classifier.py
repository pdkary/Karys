from time import time
from typing import Tuple

import numpy as np
from karys.data.ImageDataLoader import ImageDataLoader
from keras.losses import BinaryCrossentropy
from karys.models.vgg16 import Vgg16Classifier
from keras.optimizers import Adam
from karys.trainers.ImageClassificationTrainer import ImageClassificationTrainer
from keras.models import load_model, Model

image_type = ".jpg"
image_src = "./examples/discriminator/test_input/Fruit"
output_path = "./examples/discriminator/test_output"
classifier_path = output_path + "/vgg16_fruit_discriminator"

optimizer = Adam(learning_rate=4e-5)
loss = BinaryCrossentropy(from_logits=True,reduction="sum_over_batch_size")

def train(epochs, trains_per_test):
    classifier, trainer = build_classifier()
    test_loss = 0
    for i in range(epochs):
        start_time = time()
        loss = trainer.train(32, 3)
        avg_loss = np.mean(loss)

        if i % trains_per_test == 0 and i != 0:
            test_loss = trainer.test(16,1)
            avg_loss = np.mean(test_loss)
            test_output_filename = output_path + "/train-" + str(i) + ".jpg"
            trainer.save_classified_images(test_output_filename, 4,4, img_size=128)
        
        end_time = time()
        print(f"EPOCH {i}/{epochs}: loss={avg_loss}, time={end_time-start_time}")
    classifier.save(classifier_path)

def extract_features():
    classifier, trainer = build_classifier()
    features = trainer.extract_features()
    for key,val in features.items():
        np.save(output_path + "/" + key + ".npy",val)

def build_classifier() -> Tuple[Model, ImageClassificationTrainer]:
    global loss, optimizer
    data_loader = ImageDataLoader(image_src + "/",".jpg",0.1)
    try:
        print("loading classifier...")
        classifier = load_model(classifier_path)
    except Exception as e:
        classifier = Vgg16Classifier(len(data_loader.label_set)).build_graph()
    classifier.summary()
    trainer = ImageClassificationTrainer(classifier, optimizer, loss, data_loader)
    return classifier, trainer
