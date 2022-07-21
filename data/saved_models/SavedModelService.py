import json

from keras.optimizers import Optimizer
from keras.losses import Loss
from models.ModelWrapper import ModelWrapper

class SavedModelService:
    def __init__(self, model_ref_filepath):
        self.model_ref_filepath = model_ref_filepath
        self.model_refs = {}
        self.load_model_refs()

    def load_model_by_name(self,name, optimizer: Optimizer, loss: Loss) -> ModelWrapper:
        self.load_model_refs()
        filepath = self.model_refs[name]
        return ModelWrapper.load_from_filepath(filepath, optimizer, loss)

    def save_model(self, name: str, filepath: str, model_wrapper: ModelWrapper):
        self.model_refs[name] = filepath + "/" + name
        model_wrapper.save(filepath + "/" + name)
        self.save_all()

    def load_model_refs(self):
        with open(self.model_ref_filepath,'r') as model_map_file:
            self.model_refs = {**self.model_refs, **json.load(model_map_file)}
        
    def save_all(self):
        with open(self.model_ref_filepath,'w') as model_map_file:
            return json.dump(self.model_refs, model_map_file)
