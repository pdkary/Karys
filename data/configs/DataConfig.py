from abc import ABC, abstractmethod, abstractproperty

class DataConfig(ABC):

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

    # @abstractclassmethod
    # def load_from_saved_configs(cls):
    #     pass
