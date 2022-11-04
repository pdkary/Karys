import numpy as np
import tensorflow as tf

from karys.data.wrappers.ImageDataWrapper import ImageDataWrapper
from keras.models import Model
from keras.optimizers import Optimizer
from keras.losses import Loss
import keras.backend as K
def batch(ndarr, batch_size):
    N = len(ndarr)//batch_size
    return np.array(np.split(ndarr[:N*batch_size], N))

def batch_dict(adict, batch_size):
    N = len(adict)//batch_size
    batches = []
    for i in range(N):
        b_names = list(adict.keys())[i*batch_size:(i+1)*batch_size]
        b_vals = np.array(list(adict.values())[i*batch_size:(i+1)*batch_size])
        batches.append((b_names, b_vals))
    return batches

class ClassificationTrainer(object):
    def __init__(self,
                 classifier: Model,
                 optimizer: Optimizer,
                 loss: Loss,
                 labelled_input: ImageDataWrapper):
        self.classifier: Model = classifier
        self.labelled_input: ImageDataWrapper = labelled_input
        self.loss: Loss = loss
        self.optimizer: Optimizer = optimizer

        self.classifier.compile(optimizer=optimizer,loss=loss)
        self.labelled_train_data = self.labelled_input.get_train_dataset()
        self.labelled_test_data = self.labelled_input.get_validation_dataset()

        self.most_recent_output = None

    def __run_batch__(self, batch_names, batch_data, training=True):
        batch_labels = [ self.labelled_input.image_labels[n] for n in batch_names]
        vector_labels = self.labelled_input.label_vectorizer.get_label_vectors_by_category_names(batch_labels)

        features, classification_probs = self.classifier(batch_data, training=training)
        features_std = K.std(features,axie=-1)
        
        argmax_class = np.argmax(classification_probs, axis=1)
        predicted_labels = self.labelled_input.label_vectorizer.get_category_names_by_ids(argmax_class)
        self.most_recent_output = list(zip(batch_data, batch_labels, predicted_labels))

        content_loss = self.loss(vector_labels, classification_probs)
        style_loss = self.loss(np.ones_like(features_std), features_std)
        return 0.9*content_loss + 0.1*style_loss 

    def train(self, batch_size, num_batches):
        loss_bucket = []
        batched_input = batch_dict(self.labelled_train_data, batch_size)

        np.random.shuffle(batched_input)
        input_dataset = batched_input[:num_batches]

        for batch_names, batch_data in input_dataset:
            with tf.GradientTape() as grad_tape:
                loss = self.__run_batch__(batch_names, batch_data, training=True)
                loss_bucket.append(loss)
                
                classifier_grads = grad_tape.gradient(loss, self.classifier.trainable_variables)
                self.optimizer.apply_gradients(zip(classifier_grads, self.classifier.trainable_variables))
        return np.sum(loss_bucket, axis=0)
    
    def test(self, batch_size, num_batches):
        loss_bucket = []
        batched_input = batch_dict(self.labelled_test_data, batch_size)

        np.random.shuffle(batched_input)
        input_dataset = batched_input[:num_batches]

        for batch_names, batch_data in input_dataset:
            loss = self.__run_batch__(batch_names, batch_data, training=False)
            loss_bucket.append(loss)
        return np.sum(loss_bucket, axis=0)
        