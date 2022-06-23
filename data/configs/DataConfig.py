

from abc import ABC, abstractclassmethod, abstractmethod, abstractproperty
from dataclasses import dataclass

@dataclass
class DataConfig(ABC):
    batch_size: int
    num_batches: int

    @abstractproperty
    def input_shape(self):
        pass

    @abstractproperty
    def output_shape(self):
        pass

    @abstractmethod
    def to_json(self):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractclassmethod
    def load_from_saved_configs(cls):
        pass
