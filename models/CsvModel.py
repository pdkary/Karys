import numpy as np
import tensorflow as tf
from pandas import DataFrame, DatetimeIndex
from data.loaders import CsvDataLoader
from data.test_results.ModelTestResult import ModelTestResult
from data.wrappers.CsvDataWrapper import CsvDataWrapper
from data.configs.CsvDataConfig import CsvDataConfig
from data.test_results.CsvModelTestResult import CsvModelTestResult

from models.bases.ModelBase import ModelBase


def convert_rows_to_columns(row_list):
    column_list = [[] for x in row_list[0]]
    for row in row_list:
        for j, col in enumerate(row):
            column_list[j].append(col)
    return np.array(column_list)


def convert_batch_to_columns(batch):
    return np.array([convert_rows_to_columns(row_list) for row_list in batch])


class CsvModel(ModelBase):
    def set_dataconfig(self, dataconfig: CsvDataConfig):
        self.dataconfig = dataconfig

    def set_test_datawrapper(self, test_datawrapper: CsvDataWrapper):
        self.test_datawrapper = test_datawrapper
        self.test_dataset = None

    def set_train_datawrapper(self, train_datawrapper: CsvDataWrapper):
        self.train_datawrapper = train_datawrapper
        self.train_dataset = None

    def get_test_dataset(self) -> tf.data.Dataset:
        if self.test_datawrapper is None or self.dataconfig is None:
            raise ValueError(
                "test_datawrapper and dataconfig must be set before getting test dataset")
        else:
            if self.test_dataset is None:
                self.test_index, self.test_dataset = CsvDataLoader.get_horizon_dataset(
                    self.test_datawrapper.data, self.dataconfig.to_testing_config())
        return self.test_dataset

    def get_train_dataset(self) -> tf.data.Dataset:
        if self.train_datawrapper is None or self.dataconfig is None:
            raise ValueError(
                "train_datawrapper and dataconfig must be set before getting train dataset")
        else:
            if self.train_dataset is None:
                self.train_index, self.train_dataset = CsvDataLoader.get_horizon_dataset(
                    self.train_datawrapper.data, self.dataconfig)
        return self.train_dataset

    def train(self):
        train_result: ModelTestResult = super().train()
        csv_train_result = CsvModelTestResult(train_result.loss,train_result.inputs,train_result.labels,train_result.preds,self.train_index,self.dataconfig)
        return csv_train_result.to_dataframe()

    def evaluate_train_loss(self, model_input, labels, output):
        return self.loss(labels, output)

    def test(self):
        test_result: ModelTestResult = super().test()
        return CsvModelTestResult(test_result.loss,test_result.inputs,test_result.labels,test_result.preds,self.test_index,self.dataconfig.to_testing_config)

    def evaluate_test_loss(self, model_input, labels, output):
        return self.loss(labels, output)
    
    def propagate_data(self, data_wrapper: CsvDataWrapper, steps_forward: int) -> CsvDataWrapper:
        data_wrapper = data_wrapper.copy()
        for i in range(steps_forward):
            last_index, last_dataset = CsvDataLoader.get_last_horizon(data_wrapper.data, self.dataconfig)
            for in_val, label in last_dataset.as_numpy_iterator():
                next_output = self.model(in_val, training=False).numpy()[0]
                next_output_data = {c:[next_output[0][j]] for j,c in enumerate(self.dataconfig.output_columns)}

                next_output_index = DatetimeIndex([last_index[-1] + data_wrapper.time_interval])
                next_output_df = DataFrame(next_output_data, index=next_output_index)
                next_output_df.sort_index(inplace=True)
                data_wrapper.append_and_refresh_indicators(next_output_df)
        return data_wrapper[-steps_forward:]
