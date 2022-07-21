import numpy as np
from pandas import DataFrame, DatetimeIndex
import tensorflow as tf

from data.configs.CsvDataConfig import CsvDataConfig
from data.wrappers.CsvDataWrapper import CsvDataWrapper
from models.ModelWrapper import ModelWrapper

class CsvTrainer():
    def __init__(self,
                 model_base: ModelWrapper,
                 train_datawrapper: CsvDataWrapper, 
                 data_config: CsvDataConfig):
        self.model_base = model_base
        self.train_datawrapper = train_datawrapper
        self.data_config = data_config

        self.train_index_map = None
        self.train_dataset = None
        self.validation_dataset = None
    
    def get_validation_dataset(self, batch_size):
        if self.validation_dataset is None:
            self.validation_dataset = self.train_datawrapper.get_validation_dataset().batch(batch_size)
        return self.validation_dataset

    def get_train_dataset(self, batch_size):
        if self.train_dataset is None:
            self.train_dataset = self.train_datawrapper.get_train_dataset().batch(batch_size)
        return self.train_dataset
    
    def train(self, batch_size, num_batches):
        running_loss = None
        dataset = self.get_train_dataset(batch_size).shuffle(buffer_size=512).take(num_batches)
        for inputs, labels in list(dataset.as_numpy_iterator()):
            if inputs.shape[0] == batch_size:
                with tf.GradientTape() as grad_tape:
                    outputs = self.model_base.model(inputs,training=True)
                    losses = self.model_base.loss(labels, outputs)
                    running_loss = losses if running_loss is None else running_loss + losses
                    grads = grad_tape.gradient(losses, self.model_base.model.trainable_variables)
                    self.model_base.optimizer.apply_gradients(zip(grads, self.model_base.model.trainable_variables))
        return np.sum(running_loss) if running_loss is not None else 0
    
    def test(self, batch_size, num_batches):
        running_loss = None
        dataset = self.get_validation_dataset(batch_size).take(num_batches)
        for inputs, labels in list(dataset.as_numpy_iterator()):
            outputs = self.model_base.model(inputs,training=False)
            losses = self.model_base.loss(labels, outputs)
            running_loss = losses if running_loss is None else running_loss + losses
        return np.sum(running_loss) if running_loss is not None else 0

    def test_against(self, test_data_wrapper: CsvDataWrapper):
        test_dataset = test_data_wrapper.get_train_dataset().batch(1)
        out_df = []
        for inputs, labels in list(test_dataset.as_numpy_iterator()):
            outputs = self.model_base.model(inputs,training=False)
            input_row = inputs[0][-1]
            row = [*input_row, *labels[0], *outputs[0].numpy()]
            out_df.append(row)
        
        result_columns = [f"output_{n}" for n in self.data_config.target_column_names]
        columns = [*self.data_config.input_columns,*self.data_config.target_column_names, *result_columns]
        return DataFrame(columns=columns,data=out_df)
    
    def propagate_data(self, data_wrapper: CsvDataWrapper, steps_forward: int) -> CsvDataWrapper:
        data_wrapper = data_wrapper.copy()
        for i in range(steps_forward):
            last_index, last_dataset = data_wrapper.get_last_horizon()
            for in_val, label in last_dataset.as_numpy_iterator():
                next_output = self.model_base.model(in_val, training=False).numpy()[0]
                next_output_data = {c:[next_output[0][j]] for j,c in enumerate(self.data_config.output_columns)}

                next_output_index = DatetimeIndex([last_index[-1] + data_wrapper.time_interval])
                next_output_df = DataFrame(next_output_data, index=next_output_index)
                next_output_df.sort_index(inplace=True)
                data_wrapper.append_and_refresh_indicators(next_output_df)
        return data_wrapper