from pandas import DataFrame

from data.loaders import CsvDataLoader
from data.configs.CsvDataConfig import CsvDataConfig


class CsvDataWrapper(object):
    def __init__(self, df: DataFrame, data_config: CsvDataConfig):
        self.data: DataFrame = df
        self.data_config: CsvDataConfig = data_config

    @classmethod
    def load_from_file(cls, filename, data_config: CsvDataConfig):
        df = CsvDataLoader.load_data(filename, data_config)
        return cls(df, data_config)

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

    def to_csv(self, filename=None):
        self.data.to_csv(filename)

    def append_and_refresh_indicators(self, data: DataFrame):
        self.data = self.data.append(data)
        self.data.sort_index(inplace=True)
        self.data = CsvDataLoader.update_calculated_if_missing(self.data, self.data_config)
    
    def get_horizoned_data(self, data_config: CsvDataConfig):
        return CsvDataLoader.get_horizon_dataset(self.data, data_config)
