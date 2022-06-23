from abc import ABC, abstractclassmethod, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Input, Layer, Reshape
from tensorflow.keras.losses import Loss
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Optimizer

from data.test_results.ModelTestResult import ModelTestResult


@dataclass
class ModelBase(ABC):
    input_shape: Tuple
    output_shape: Tuple
    layers: List[Layer]
    optimizer: Optimizer
    loss: Loss

    flatten_input: bool = True
    model_filepath: str = None
    
    # @abstractclassmethod
    # def load_from_saved_models(cls):
    #     pass

    @abstractmethod
    def get_test_dataset(self) -> tf.data.Dataset:
        pass

    @abstractmethod
    def get_train_dataset(self) -> tf.data.Dataset:
        pass

    @abstractmethod
    def evaluate_train_loss(self, model_input, labels, model_output):
        return self.loss(labels, model_output)

    @abstractmethod
    def evaluate_test_loss(self, model_input, labels, model_output):
        return self.loss(labels, model_output)

    @property
    def layer_sizes(self):
        return [x.input_spec.shape for x in self.layers]

    def train(self) -> ModelTestResult:
        running_loss = 0.0
        inputs, labels, preds = [], [], []
        dataset = self.get_train_dataset()
        for model_input, input_labels in list(dataset.as_numpy_iterator()):
            with tf.GradientTape() as grad_tape:
                inputs.append(model_input)
                labels.append(input_labels)
                output = self.model(model_input, training=True)
                preds.append(output)
                loss = self.evaluate_train_loss(model_input, input_labels, output)
                running_loss += np.sum(loss)
                
                grads = grad_tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return ModelTestResult(running_loss, inputs, labels, preds)

    def test(self) -> ModelTestResult:
        running_loss = 0.0
        inputs, labels, preds = [], [], []
        dataset = self.get_test_dataset()
        for model_input, model_labels in list(dataset.as_numpy_iterator()):
            inputs.append(model_input)
            labels.append(model_labels)
            output = self.model(model_input, training=False)
            preds.append(output)
            loss = self.evaluate_test_loss(model_input, model_labels, output)
            running_loss += np.sum(loss)
        return ModelTestResult(running_loss, inputs, labels, preds)

    def build(self, name: str = None, silent=False):
        if self.model_filepath is not None:
            self.model: Model = load_model(self.model_filepath)
            self.input_layer = Input(shape=self.model.input_shape)
        else:
            self.input_layer = Input(shape=self.input_shape)
            functional_model = self.input_layer

            if self.flatten_input:
                functional_model = Flatten()(self.input_layer)

            for d in self.layers:
                functional_model = d(functional_model)

            functional_model = Reshape(target_shape=self.output_shape)(functional_model)
            self.model = Model(
                self.input_layer, outputs=functional_model, name=name)
            self.model.compile(optimizer=self.optimizer, loss=self.loss)

        if not silent:
            self.model.summary()
        return self.model

    def save(self, filepath):
        self.model.save(filepath)

    def to_json(self):
        return dict(input_shape=self.input_shape,
                    output_shape=self.output_shape,
                    layers=[l.get_config() for l in self.layers],
                    optimizer=str(self.optimizer._name),
                    loss=str(self.loss.__name__))

    def __str__(self):
        return str(self.to_json())
