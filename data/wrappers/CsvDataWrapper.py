from collections import deque
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing

from pandas import DataFrame

from data.configs.CsvDataConfig import CsvDataConfig
from data.wrappers.DataWrapper import DataWrapper

def index_to_datetime(df: pd.DataFrame, index_name: str):
    assert index_name in df.columns
    if df[index_name].dtype == object:
        df[index_name] = pd.to_datetime(df[index_name], format='%Y-%m-%d %H:%M:%S')
    else:
        df[index_name] = pd.to_datetime(df[index_name], unit='ms')
    df.set_index(index_name, inplace=True)
    return df

class CsvDataWrapper(DataWrapper):
    def __init__(self, df: DataFrame, data_config: CsvDataConfig, validation_percentage = 0.05):
        super(CsvDataWrapper, self).__init__(data_config, validation_percentage)
        self.data: DataFrame = df
        self.data_config: CsvDataConfig = data_config
        self.add_or_update_calculated_columns()
        self.add_or_update_target_columns()

    @classmethod
    def load_from_file(cls, filename, data_config: CsvDataConfig, validation_percentage = 0.05):
        data = pd.read_csv(filename)
        data = index_to_datetime(data, data_config.index_name)[data_config.input_columns]
        return cls(data, data_config, validation_percentage)
    
    def add_or_update_calculated_columns(self):
        for cc in self.data_config.calculated_columns:
            if len(cc.input_columns) == 1:
                if cc.input_columns[0] == self.data_config.index_name:
                    indicator_data = self.data.index.to_numpy().flatten()
                else:
                    indicator_data = self.data.loc[:,cc.input_columns].to_numpy().flatten()
            else:
                indicator_data = self.data.loc[:, cc.input_columns].to_numpy()
            
            new_indicator_column = cc.func(indicator_data, *list(cc.parameters.values()))
            self.data[cc.name] = new_indicator_column
            self.data[cc.name].dropna(inplace=True)
        self.data.dropna(inplace=True)
    
    def add_or_update_target_columns(self):
        target_columns = self.data_config.target_column_names
        self.data[target_columns] = self.data[self.data_config.output_columns].shift(-self.data_config.lookahead)

    @property
    def calculated_column_names(self):
        return [x.name for x in self.data_config.calculated_columns]

    @property
    def time_interval(self):
        return self.data.index[1] - self.data.index[0]
    
    # ## override so we can access dataframe directly
    def __getitem__(self, key):
        # check if we want multiple rows, or just one
        if isinstance(key, tuple):
            return self.data.loc[[k for k in key]]
        # useful for grabbing subsets
        elif isinstance(key, slice):
            return CsvDataWrapper(self.data.iloc[key], self.data_config)
        else:
            return self.data[key]

    def __len__(self):
        return len(self.data)

    def copy(self):
        return CsvDataWrapper(self.data.copy(),self.data_config)
    
    def find_by_input_value(self,partial_input):
        df = self.data.loc[:,self.data_config.input_columns].pct_change()
        df.replace([-np.inf,np.inf],np.nan, inplace=True)
        df.dropna(inplace=True)
        queries = [f'`{c}` == {partial_input[i]}' for i,c in enumerate(self.data_config.input_columns)]
        query_str = ' & '.join(queries)
        return df.query(query_str)

    def to_csv(self, filename=None):
        self.data.to_csv(filename)

    def append_and_refresh_indicators(self, data: DataFrame):
        self.data = self.data.append(data)
        self.data.sort_index(inplace=True)
        self.add_or_update_calculated_columns()
    
    def process_data(self, df: DataFrame):
        for col in [*self.data_config.input_columns, *self.data_config.target_column_names]:
            df[col] = df[col].pct_change()
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)
        
        horizon_data = []  # this is a list that will CONTAIN the sequences
        target_data = []
        horizon = deque(maxlen=self.data_config.horizon)
        T = len(self.data_config.target_column_names)
        for row in df.values:  # iterate over the values
            horizon.append([n for n in row[:-T]])  # store all but the target
            if len(horizon) == self.data_config.horizon:  # make sure we have 60 sequences!
                horizon_data.append(np.array(horizon))
                target_data.append(row[-T:])
        return tf.data.Dataset.from_tensor_slices((horizon_data, target_data))

    def get_train_dataset(self):
        df = self.data[:-int(len(self.data)*self.validation_percentage)]
        return self.process_data(df)
    
    def get_validation_dataset(self):
        df = self.data[-int(len(self.data)*self.validation_percentage):]
        return self.process_data(df)
    
    def get_last_horizon(self):
        dataset = self.get_validation_dataset
        return list(dataset.batch(1).take(1))[-1]

