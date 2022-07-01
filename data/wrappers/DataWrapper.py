

from abc import ABC, abstractclassmethod, abstractmethod

from data.configs.DataConfig import DataConfig


class DataWrapper(ABC):
    def __init__(self, data_config: DataConfig, train_test_ratio = 0.7):
        self.data_config = data_config
        self.train_test_ratio = train_test_ratio

    @abstractclassmethod
    def load_from_file(cls, filename: str, data_config: DataConfig):
        pass

    @abstractmethod
    def get_train_dataset(self):
        pass

    @abstractmethod
    def get_test_dataset(self):
        pass


