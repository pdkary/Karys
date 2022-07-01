from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple
from data.configs.DataConfig import DataConfig
from data.saved_models.SavedModelService import SavedModelService, TrainedModelReference

from models.ModelWrapper import ModelWrapper

@dataclass
class MultiModelWrapper(object):
    model_set: Dict[str,TrainedModelReference]

    @abstractmethod
    def train(self):
        pass
    
    @property
    def model_sets(self) -> List[Tuple[ModelWrapper, DataConfig]]:
        return list(self.model_set.items())
    
    @classmethod
    def load_from_saved_models(cls):
        model_data = SavedModelService.load_from_file()
        return cls(model_data)
    
    def save(self):
        SavedModelService.update(self.model_set)
