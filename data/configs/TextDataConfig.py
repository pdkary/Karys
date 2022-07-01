from dataclasses import dataclass
from typing import Tuple

from data.configs.DataConfig import DataConfig


@dataclass
class TextDataConfig(DataConfig):
    num_words: int
    sentence_length : int
    input_sentences: int
    predicted_sentences: int = 1
    ignore_output: bool = False

    @property
    def input_shape(self) -> Tuple: 
        return (self.input_sentences, self.sentence_length)
    
    @property
    def output_shape(self) -> Tuple:
        if self.ignore_output:
            return [self.sentence_length]
        return (self.predicted_sentences, self.sentence_length)
    
    def __str__(self):
        return str(self.to_json())

    def to_json(self):
        return dict(num_words=self.num_words,
                    sentence_length=self.sentence_length, 
                    input_sentences=self.input_sentences, 
                    predicted_sentences=self.predicted_sentences, 
                    ignore_output=self.ignore_output)
