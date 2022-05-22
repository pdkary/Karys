from typing import List

import numpy as np
import tensorflow as tf
from data.CsvDataConfig import CsvDataConfig, CalculatedColumnConfig
from pandas import DataFrame, read_csv, to_datetime


def load_data(filename: str, index_name: str, calculated_columns: List[CalculatedColumnConfig] = []) -> DataFrame:
    data = read_csv(filename)
    assert index_name in data.columns
    if data[index_name].dtype == object:
        data[index_name] = to_datetime(
            data[index_name], format='%Y-%m-%d %H:%M:%S')
    else:
        data[index_name] = to_datetime(data[index_name], unit='ms')
    data.set_index(index_name, inplace=True)
    data = update_calculated_if_missing(data, calculated_columns)
    data.to_csv(filename)
    return data


def update_calculated_if_missing(df: DataFrame, calculated_columns: List[CalculatedColumnConfig]):
    new_CCs = [c for c in calculated_columns if c.name not in df.columns]
    for cc in new_CCs:
        if len(cc.input_columns) == 1:
            if cc.input_columns[0] == df.index.name:
                indicator_data = df.index.to_numpy().flatten()
            else:
                indicator_data = df.loc[:,
                                        cc.input_columns].to_numpy().flatten()
        else:
            indicator_data = df.loc[:, cc.input_columns].to_numpy()
        new_indicator_column = cc.func(
            indicator_data, *list(cc.parameters.values()))
        df[cc.name] = new_indicator_column
        df[cc.name] = df[cc.name].ffill()
    return df


def get_columns_and_calculateds(df: DataFrame, columns: List[str], calculated_columns: List[CalculatedColumnConfig]):
    df = update_calculated_if_missing(df, calculated_columns)
    return df[[*columns, *list([i.name for i in calculated_columns])]]


def get_columns_and_null_calculateds(df: DataFrame, columns: List[str], calculated_columns: List[CalculatedColumnConfig]):
    copy_df = df.copy()
    for c in calculated_columns:
        copy_df[c.name] = float('nan')
    return copy_df[[*columns, *list([c.name for c in calculated_columns])]]


def get_horizon_dataset(df: DataFrame, data_ref: CsvDataConfig) -> tf.data.Dataset:
    if data_ref.null_calculated:
        filtered_df = get_columns_and_null_calculateds(
            df, data_ref.data_columns, data_ref.calculated_column_configs)
    else:
        filtered_df = get_columns_and_calculateds(
            df, data_ref.data_columns, data_ref.calculated_column_configs)
    # we want to get every possible <horizon> length subset of the list, ending <lookahead> distance from the end
    # we also want to be sure that our <labels> have shape (<lookahead>, ... )
    # we also want to preserve then end of the list over the beginninng
    length = len(filtered_df)
    H = data_ref.horizon
    L = data_ref.lookahead
    end = length - L - H
    n_list = list(range(end))
    index = np.array([filtered_df.index[i+H] for i in n_list])
    examples = np.array([filtered_df.iloc[i:i+H][data_ref.input_columns]
                         for i in n_list], dtype=np.float32)
    labels = np.array([filtered_df.iloc[i+H:i+H+L][data_ref.output_columns]
                       for i in n_list], dtype=np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((examples, labels))
    return index, dataset.batch(data_ref.batch_size)


def get_last_horizon(df: DataFrame, data_ref: CsvDataConfig) -> tf.data.Dataset:
    if data_ref.null_calculated:
        filtered_df = get_columns_and_null_calculateds(
            df, data_ref.data_columns, data_ref.calculated_column_configs)
    else:
        filtered_df = get_columns_and_calculateds(
            df, data_ref.data_columns, data_ref.calculated_column_configs)
    length = len(filtered_df)
    H = data_ref.horizon
    L = data_ref.lookahead
    end = length - L - H
    examples = np.array([filtered_df.iloc[end: end+H]
                         [data_ref.input_columns]], dtype=np.float32)
    labels = np.array([filtered_df.iloc[end+H: end + H + L]
                       [data_ref.output_columns]], dtype=np.float32)
    final_index = filtered_df.index[end+H: end + H + L]
    dataset = tf.data.Dataset.from_tensor_slices((examples, labels))
    return final_index, dataset.batch(1).take(1)
