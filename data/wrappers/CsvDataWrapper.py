from typing import Dict, List

from pandas import DataFrame

from data.loaders import CsvDataLoader
from data.configs.CsvDataConfig import CalculatedColumnConfig


class CsvDataWrapper(object):
    def __init__(self, df: DataFrame, calculated_cols_by_name: Dict[str, CalculatedColumnConfig] = {}):
        self.data: DataFrame = df
        self.calculated_columns_configs: Dict[str,CalculatedColumnConfig] = calculated_cols_by_name

    @classmethod
    def load_from_file(cls, filename, index_name, indicators: List[CalculatedColumnConfig]):
        df = CsvDataLoader.load_data(filename, index_name, indicators)
        return cls(df, {i.name: i for i in indicators})

    @property
    def calculated_columns(self):
        return list(self.calculated_columns_configs.keys())

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
            return CsvDataWrapper(self.data.iloc[key], self.calculated_columns_configs)
        else:
            return self.data[key]

    def __len__(self):
        return len(self.data)

    def copy(self):
        return CsvDataWrapper(self.data.copy(), self.calculated_columns_configs)

    def to_csv(self, filename=None):
        self.data.to_csv(filename)

    def set_indicators(self, new_indicators: List[CalculatedColumnConfig]):
        self.data = CsvDataLoader.update_calculated_if_missing(
            self.data, new_indicators)
        self.calculated_columns_configs = {i.name: i for i in new_indicators}

    def append_and_refresh_indicators(self, data: DataFrame):
        self.data = self.data.append(data)
        self.data.sort_index(inplace=True)
        self.set_indicators(list(self.calculated_columns_configs.values()))
