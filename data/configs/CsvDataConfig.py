from copy import copy
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from data.configs.DataConfig import DataConfig
from data.saved_models import SavedModelService


@dataclass
class CalculatedColumnConfig(object):
    name: str
    func: Callable
    input_columns: List[str]
    parameters: Dict[str, str]

    def __str__(self):
        return str(self.to_json())

    def to_json(self):
        return dict(name=self.name, func=self.func.__name__, input_columns=self.input_columns, parameters=self.parameters)


@dataclass
class CsvDataConfig(DataConfig):
    data_columns: List[str]
    calculated_column_configs: List[CalculatedColumnConfig]
    horizon: int
    lookahead: int
    null_calculated: bool = False
    predict_calculated: bool = False

    @property
    def input_shape(self) -> Tuple:
        return (self.horizon, len(self.input_columns))

    @property
    def output_shape(self) -> Tuple:
        if self.predict_calculated:
            return (self.lookahead, len(self.input_columns))
        else:
            return (self.lookahead, len(self.data_columns))

    @property
    def input_columns(self) -> List[str]:
        return [*self.data_columns, *self.calculated_columns]

    @property
    def output_columns(self) -> List[str]:
        if self.predict_calculated:
            return [*self.data_columns, *self.calculated_columns]
        else:
            return self.data_columns

    @property
    def calculated_columns(self) -> List[str]:
        return [c.name for c in self.calculated_column_configs]

    @property
    def calculated_column_str(self) -> str:
        return "Calculated Columns=[" + ", ".join(self.calculated_columns) + "]"

    def __str__(self):
        return str(self.to_json())

    def to_json(self):
        return dict(data_columns=self.data_columns,
                    calculated_column_configs=[i.to_json()
                                for i in self.calculated_column_configs],
                    horizon=self.horizon,
                    lookahead=self.lookahead,
                    null_calculated=self.null_calculated)

    def as_null_indicators(self):
        new_data_ref = copy(self)
        new_data_ref.null_calculated = True
        return new_data_ref

    def to_testing_config(self):
        new_data_ref = copy(self)
        return new_data_ref
    
    @classmethod
    def load_from_saved_configs(cls, filepath, calc_columns: List[CalculatedColumnConfig]):
        get_indicator = lambda x: [i for i in calc_columns if i.name == x][0]
        json_dict = SavedModelService.get_data_reference_dict()[filepath]
        json_dict['calculated_column_configs'] = [get_indicator(x) for x in json_dict['calculated_column_configs']]
        return cls(**json_dict)
