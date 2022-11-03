

from data.configs.ImageDataConfig import ImageDataConfig
from data.wrappers.ImageDataWrapper import ImageDataWrapper
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
from keras.models import load_model
from models.vgg16 import ReverseVgg16Generator, Vgg16Classifier
from time import time
import numpy as np
import dictdiffer

from trainers.ClassificationTrainer import ClassificationTrainer                                          

def test_train_classification_model(classifier, optimizer, loss, data_wrapper, epochs=5, trains_per_test=4):
    trainer = ClassificationTrainer(classifier, optimizer, loss, data_wrapper)
    output_path = "./examples/discriminator/test_output"

    test_loss = 0
    for i in range(epochs):
        start_time = time()
        train_loss = trainer.train(4, 1)
        avg_loss = np.mean(train_loss)

        if i % trains_per_test == 0 and i != 0:
            test_loss = trainer.test(16,1)
            avg_loss = np.mean(test_loss)
            test_output_filename = output_path + "/train-" + str(i) + ".jpg"
            data_wrapper.save_classified_images(test_output_filename, trainer.most_recent_output, img_size=32)
        
        end_time = time()
        print(f"EPOCH {i}/{epochs}: loss={avg_loss}, time={end_time-start_time}")

def test_classification_model():
    ##set up data
    base_path = "./examples/discriminator/test_input/Fruit"
    extra_labels = ["generated"]
    image_config = ImageDataConfig(image_shape=(224,224, 3),image_type=".jpg", preview_rows=4, preview_cols=4, load_n_percent=20)
    data_wrapper = ImageDataWrapper.load_from_labelled_directories(base_path + '/', image_config, extra_labels, validation_percentage=0.1)

    ## Build Model
    classifier = Vgg16Classifier(len(data_wrapper.classification_labels)).build_graph()
    classifier.summary()

    optimizer = Adam(learning_rate=4e-5)
    loss = BinaryCrossentropy(from_logits=True,reduction="sum_over_batch_size")

    ## Train model
    test_train_classification_model(classifier, optimizer, loss, data_wrapper, 10, 9)

    ## Save model
    output_path = "./examples/discriminator/test_output"
    classifier.save(output_path + "/vgg16_fruit_classifier")

    # Reload model
    new_classifier = load_model(output_path + "/vgg16_fruit_classifier")

    ## Train model again
    test_train_classification_model(new_classifier, optimizer, loss, data_wrapper,10, 9)
    
    ## Save model again
    new_classifier.save(output_path + "/vgg16_fruit_classifier")
