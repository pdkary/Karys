import numpy as np
import tensorflow as tf
from karys.data.ImageDataLoader import ImageDataLoader
from matplotlib import pyplot as plt
from PIL import Image

from keras.models import Model
from keras.optimizers import Optimizer
from keras.losses import Loss
import keras.backend as K

def scale_to_255(img):
    img_min = np.min(img)
    img_max = np.max(img)
    return 255*(img - img_min)/(img_max - img_min + 1e-8)

class ImageClassificationTrainer(object):
    def __init__(self,
                 classifier: Model,
                 optimizer: Optimizer,
                 loss: Loss,
                 labelled_input: ImageDataLoader):
        self.classifier: Model = classifier
        self.labelled_input: ImageDataLoader = labelled_input
        self.loss: Loss = loss
        self.optimizer: Optimizer = optimizer

        self.classifier.compile(optimizer=optimizer,loss=loss)
        self.most_recent_output = None
        self.image_shape = classifier.input_shape[-3:-1]

    def __run_batch__(self, batch_labels, batch_data, training=True):
        vector_labels = self.labelled_input.get_label_vectors_by_names(batch_labels)

        classification_probs = self.classifier(batch_data, training=training)
        
        argmax_class = np.argmax(classification_probs, axis=1)
        predicted_labels = self.labelled_input.get_label_vectors_by_ids(argmax_class)
        self.most_recent_output = list(zip(batch_data, batch_labels, predicted_labels))

        content_loss = self.loss(vector_labels, classification_probs)
        return content_loss

    def train(self, batch_size, num_batches):
        loss_bucket = []
        batched_input = self.labelled_input.get_sized_training_batch_set(num_batches, batch_size, self.image_shape)

        for batch_labels, batch_data in batched_input:
            with tf.GradientTape() as grad_tape:
                loss = self.__run_batch__(batch_labels, batch_data, training=True)
                loss_bucket.append(loss)
                
                classifier_grads = grad_tape.gradient(loss, self.classifier.trainable_variables)
                self.optimizer.apply_gradients(zip(classifier_grads, self.classifier.trainable_variables))
        return np.sum(loss_bucket, axis=0)
    
    def test(self, batch_size, num_batches):
        loss_bucket = []
        batched_input = self.labelled_input.get_sized_validation_batch_set(num_batches, batch_size, self.image_shape)

        for batch_names, batch_data in batched_input:
            loss = self.__run_batch__(batch_names, batch_data, training=False)
            loss_bucket.append(loss)
        return np.sum(loss_bucket, axis=0)
    
    def save_classified_images(self, filename, preview_rows, preview_cols, img_size = 32):
        image_shape = (img_size, img_size)
        channels = image_shape[-1]

        fig,axes = plt.subplots(preview_rows,preview_cols, figsize=image_shape)

        for row in range(preview_rows):
            for col in range(preview_cols):
                img, label, pred = self.most_recent_output[row*preview_rows+col]
                pass_fail = "PASS" if np.all(pred == label) else "FAIL"
                text_label = pred + " || " + label + " || " + pass_fail
                img = scale_to_255(img)

                if channels == 1:
                    img = np.reshape(img,newshape=image_shape)
                else:
                    img = np.array(img)
                    img = Image.fromarray((img).astype(np.uint8))
                    img = img.resize(image_shape, Image.BICUBIC)
                    img = np.asarray(img)
                    
                axes[row,col].imshow(img)
                axes[row,col].set_title(text_label, fontsize=img_size//2)

        fig.savefig(filename)
        plt.close()
        
    def extract_features(self):
        features_by_name = {}
        for k,v in self.labelled_input.get_all_images_sized(self.classifier.input_shape[1:-1]).items():
            key_features = []
            print("Extracting Features from " + k,end="")
            for i,img in enumerate(v):
                if i%10 == 0:
                    print(".",end="")
                features, classifications = self.classifier(np.array([img]), training=False)
                key_features.append(features)
            print("Done")
            feature_stack = np.concatenate(key_features)
            features_by_name[k] = feature_stack
        return features_by_name
