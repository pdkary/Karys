from time import time

import numpy as np
from karys.data.configs.ImageDataConfig import ImageDataConfig
from karys.data.wrappers.ImageDataWrapper import ImageDataWrapper
from keras.losses import BinaryCrossentropy
from karys.models.vgg16 import Vgg16Classifier
from keras.optimizers import Adam
from trainers.ClassificationTrainer import ClassificationTrainer
from keras.models import load_model

image_type = ".jpg"
image_src = "./examples/discriminator/test_input/Fruit"
output_path = "./examples/discriminator/test_output"
classifier_path = output_path + "/vgg16_fruit_discriminator"

optimizer = Adam(learning_rate=4e-5)
loss = BinaryCrossentropy(from_logits=True,reduction="sum_over_batch_size")

def train(epochs, trains_per_test):
    data_wrapper, classifier = build_classifier()
    optimizer = Adam(learning_rate=4e-5)
    loss = BinaryCrossentropy(from_logits=True,reduction="sum_over_batch_size")
    trainer = ClassificationTrainer(classifier, optimizer, loss, data_wrapper)
    test_loss = 0
    for i in range(epochs):
        start_time = time()
        loss = trainer.train(32, 3)
        avg_loss = np.mean(loss)

        if i % trains_per_test == 0 and i != 0:
            test_loss = trainer.test(16,1)
            avg_loss = np.mean(test_loss)
            test_output_filename = output_path + "/train-" + str(i) + ".jpg"
            data_wrapper.save_classified_images(test_output_filename, trainer.most_recent_output, img_size=32)
        
        end_time = time()
        print(f"EPOCH {i}/{epochs}: loss={avg_loss}, time={end_time-start_time}")
    classifier.save(classifier_path)

def build_classifier():
    image_config = ImageDataConfig(image_shape=(224,224, 3),image_type=image_type, preview_rows=4, preview_cols=4, load_n_percent=100)
    data_wrapper = ImageDataWrapper.load_from_labelled_directories(image_src + '/', image_config, ['Not Fruit'], validation_percentage=0.1)
    classification_labels = list(set(data_wrapper.image_labels.values()))
    
    try:
        print("loading classifier...")
        classifier = load_model(classifier_path)
    except Exception as e:
        classifier = Vgg16Classifier(len(classification_labels)+1).build_graph()
    classifier.summary()
    return data_wrapper, classifier
