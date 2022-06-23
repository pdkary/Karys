
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from pandas import DataFrame

from data.configs.CsvDataConfig import CsvDataConfig
from data.test_results.ModelTestResult import ModelTestResult

def convert_rows_to_columns(row_list):
    column_list = [[] for x in row_list[0]]
    for row in row_list:
        for j, col in enumerate(row):
            column_list[j].append(col)
    return column_list

def convert_rowset_to_columnset(rowset, column_labels: List[str]):
    columnset = {cl: [] for cl in column_labels}
    for row in rowset:
        for j, v in enumerate(row):
            columnset[column_labels[j]].append(v)
    return columnset

def pad_columnset(columnset: Dict[str, List[np.float32]]):
    max_length = max([len(k) for k in columnset.values()])
    def pad_column(x): return [float('nan')]*(max_length - len(x)) + x
    return {k: pad_column(v) for k, v in columnset.items()}

@dataclass
class CsvModelTestResult(ModelTestResult):
    index: List
    data_ref: CsvDataConfig

    @property
    def input_columns(self):
        return self.data_ref.input_columns

    @property
    def output_columns(self):
        return self.data_ref.output_columns

    def to_dataframe(self):
        input_columnset = convert_rowset_to_columnset(
            self.inputs,  [x + "_input" for x in self.input_columns])
        label_columnset = convert_rowset_to_columnset(
            self.labels, [x + "_label" for x in self.output_columns])
        output_columnset = convert_rowset_to_columnset(
            self.preds, [x + "_pred" for x in self.output_columns])
        full_columnset = pad_columnset(
            {**input_columnset, **label_columnset, **output_columnset})
        return DataFrame(full_columnset, index=self.index)

    @property
    def is_result_garbage(self):
        column_preds = convert_rows_to_columns(self.preds)
        column_lbls = convert_rows_to_columns(self.labels)
        pred_zeros = [np.all(np.array(preds) == 0.0) for preds in column_preds]
        pred_nans = [np.all(np.array(preds) == float('nan'))
                     for preds in column_preds]
        pred_stds_too_big = [np.std(preds) > np.std(
            lbl)*2 for preds, lbl in zip(column_preds, column_lbls)]
        pred_means_too_big = [np.mean(preds) > np.mean(
            lbl)*4 for preds, lbl in zip(column_preds, column_lbls)]
        return np.any(np.array(pred_zeros)) or np.any(np.array(pred_nans)) or np.any(np.array(pred_stds_too_big)) or np.any(np.array(pred_means_too_big))
