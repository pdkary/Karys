import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

from models.bases.ModelBase import ModelBase
from data.configs.DataConfig import DataConfig

MODEL_REF_FILENAME = 'data/saved_models/saved_model_map.json'

@dataclass
class TrainedModelReference:
    filepath: str
    model: ModelBase
    dataconfig: DataConfig
    fitness: float

class SavedModelService:
    def get_by_name(self,name):
        return SavedModelService.__load__()[name]
    
    def get_by_model_base(self,model_base: ModelBase):
        refs = SavedModelService.__load__()
        refs_by_model = {v.model:v for v in refs.values()}
        return refs_by_model[model_base]

    @staticmethod
    def update(new_refs: Dict[str,TrainedModelReference]):
        refs = SavedModelService.__load__()
        refs = {**refs,**new_refs}
        SavedModelService.__save__(refs)

    @staticmethod
    def __load__():
        with open(MODEL_REF_FILENAME,'r') as model_map_file:
            return json.load(model_map_file)
    
    @staticmethod
    def __save__(refs: Dict[str,TrainedModelReference]):
        with open(MODEL_REF_FILENAME,'w') as model_map_file:
            return json.dump(refs,model_map_file)

