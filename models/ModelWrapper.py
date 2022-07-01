from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from tensorflow.keras.layers import Flatten, Input, Layer, Reshape, Dense
from tensorflow.keras.losses import Loss
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Optimizer


@dataclass
class ModelWrapper():
    input_shape: Tuple
    output_shape: Tuple
    layers: List[Layer]
    optimizer: Optimizer
    loss: Loss

    flatten_input: bool = True
    model_filepath: str = None

    @property
    def layer_sizes(self):
        return [x.input_spec.shape for x in self.layers]

    def build(self, name: str = None, silent=False):
        if self.model_filepath is not None:
            self.model: Model = load_model(self.model_filepath)
            self.input_layer = Input(shape=self.model.input_shape)
        else:
            self.input_layer = Input(shape=self.input_shape)
            functional_model = self.input_layer

            if self.flatten_input:
                functional_model = Flatten()(self.input_layer)

            for l in self.layers:
                functional_model = l(functional_model)

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
