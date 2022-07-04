from dataclasses import dataclass
from typing import Tuple

from data.configs.DataConfig import DataConfig


@dataclass
class TextDataConfig(DataConfig):
    vocab_size: int
    input_length : int

    @property
    def input_shape(self) -> Tuple: 
        return [self.input_length]
    
    @property
    def output_shape(self) -> Tuple:
        return [self.vocab_size]
    
    def __str__(self):
        return str(self.to_json())

    def to_json(self):
        return dict(vocab_size=self.vocab_size,
                    input_length=self.input_length, 
                    scale_data=self.scale_data)
