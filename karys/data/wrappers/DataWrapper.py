

from abc import ABC, abstractclassmethod, abstractmethod

from karys.data.configs.DataConfig import DataConfig


class DataWrapper(ABC):
    def __init__(self, data_config: DataConfig, validation_percentage = 0.05):
        self.data_config = data_config
        self.validation_percentage = validation_percentage

    @abstractclassmethod
    def load_from_file(cls, filename: str, data_config: DataConfig):
        pass

    @abstractmethod
    def get_train_dataset(self):
        pass

    @abstractmethod
    def get_validation_dataset(self):
        pass


