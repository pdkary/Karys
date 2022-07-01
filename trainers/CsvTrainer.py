import numpy as np
from pandas import DataFrame, DatetimeIndex
import tensorflow as tf

from data.configs.CsvDataConfig import CsvDataConfig
from data.loaders import CsvDataLoader
from data.wrappers.CsvDataWrapper import CsvDataWrapper
from models.ModelWrapper import ModelWrapper
from trainers.outputs.CsvModelOutput import CsvModelOutput

class CsvTrainer():
    def __init__(self,
                 model_base: ModelWrapper,
                 train_datawrapper: CsvDataWrapper, 
                 test_datawrapper: CsvDataWrapper,
                 data_config: CsvDataConfig):
        self.model_base = model_base
        self.train_datawrapper = train_datawrapper
        self.test_datawrapper = test_datawrapper
        self.data_config = data_config

        self.train_index_map = None
        self.train_dataset = None
        self.test_dataset = None
    
    def get_train_index(self,input_val):
        for k, v in self.train_index_map.items():
            if np.all(v == input_val):
                return k
        return None
    
    def get_test_index(self,input_val):
        for k, v in self.test_index_map.items():
            if np.all(v == input_val):
                return k
        return None
    
    def prepare_data(self, batch_size):
        self.train_index_map, train_data = self.train_datawrapper.get_horizoned_data(self.data_config)
        self.train_dataset = train_data.batch(batch_size)

        self.test_index_map, test_data = self.test_datawrapper.get_horizoned_data(self.data_config)
        self.test_dataset = test_data.batch(batch_size)
    
    def get_test_dataset(self, batch_size):
        if self.test_dataset is None:
            self.prepare_data(batch_size)
        return self.test_dataset

    def get_train_dataset(self, batch_size):
        if self.train_dataset is None:
            self.prepare_data(batch_size)
        return self.train_dataset
    
    def train(self, batch_size, num_batches) -> CsvModelOutput:
        dataset = self.get_train_dataset(batch_size).shuffle(buffer_size=512).take(num_batches)
        train_output = CsvModelOutput(self.train_datawrapper.time_interval)
        for inputs, labels in list(dataset.as_numpy_iterator()):
            with tf.GradientTape() as grad_tape:
                outputs = self.model_base.model(inputs,training=True)
                losses = self.model_base.loss(labels, outputs)
                for input,output,label,loss in zip(inputs,outputs,labels,losses):
                    index = self.get_train_index(input)
                    train_output.add(index,loss.numpy(),input,output.numpy(),label)
                grads = grad_tape.gradient(losses, self.model_base.model.trainable_variables)
                self.model_base.optimizer.apply_gradients(zip(grads, self.model_base.model.trainable_variables))
        return train_output
    
    def test(self, batch_size, num_batches) -> CsvModelOutput:
        dataset = self.get_test_dataset(batch_size).take(num_batches)
        test_output = CsvModelOutput(self.test_datawrapper.time_interval)
        for inputs, labels in list(dataset.as_numpy_iterator()):
            outputs = self.model_base.model(inputs,training=True)
            losses = self.model_base.loss(labels, outputs)
            for input,output,label,loss in zip(inputs,outputs,labels,losses):
                index = self.get_test_index(input)
                test_output.add(index,loss.numpy(),input,output.numpy(),label)
        return test_output
    
    def propagate_data(self, data_wrapper: CsvDataWrapper, steps_forward: int) -> CsvDataWrapper:
        data_wrapper = data_wrapper.copy()
        for i in range(steps_forward):
            last_index, last_dataset = CsvDataLoader.get_last_horizon(data_wrapper.data, self.data_config)
            for in_val, label in last_dataset.as_numpy_iterator():
                next_output = self.model_base.model(in_val, training=False).numpy()[0]
                next_output_data = {c:[next_output[0][j]] for j,c in enumerate(self.data_config.output_columns)}

                next_output_index = DatetimeIndex([last_index[-1] + data_wrapper.time_interval])
                next_output_df = DataFrame(next_output_data, index=next_output_index)
                next_output_df.sort_index(inplace=True)
                data_wrapper.append_and_refresh_indicators(next_output_df)
        return data_wrapper