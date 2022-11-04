from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class CategoricalLabel():
    def __init__(self, categories: List[str], flags: List[str] = None):
        flags = [] if flags is None else flags
        self.categories = categories
        self.flags = flags
        self.category_dim = len(categories)
        self.flag_dim = len(flags)
        self.label_dim = len(categories) + len(flags)

        self.labels_by_id = pd.DataFrame([*categories, *flags], columns=["categories"], index=range(self.label_dim))
        print(self.labels_by_id)
        one_hot_encoder = OneHotEncoder(sparse=False)
        one_hot_encoder.fit(self.labels_by_id)
        label_df_encoded = one_hot_encoder.transform(self.labels_by_id)
        self.label_vectors_by_name = pd.DataFrame(data=label_df_encoded, columns=one_hot_encoder.categories_)
    
    @property
    def shape(self):
        return (self.label_dim,)
    
    def get_label_id_by_category_name(self, name):
        return self.labels_by_id.loc[self.labels_by_id["categories"] == name].index[0]

    def get_label_ids_by_category_names(self, names):
        return [self.get_label_id_by_category_name(n) for n in names]

    def get_label_vector_by_category_name(self,name):
        return self.label_vectors_by_name[name]
    
    def get_label_vectors_by_category_names(self,names, flags=None):
        category_vectors = np.array([self.get_label_vector_by_category_name(n) for n in names],dtype=np.float32)[:,:,0]
        if flags is not None:
            flag_vectors = np.array([self.get_label_vector_by_category_name(f) for f in flags],dtype=np.float32)
            return category_vectors + flag_vectors
        return category_vectors
    
    def get_category_names_by_ids(self, ids):
        return self.labels_by_id['categories'].iloc[ids]
    
    def ones(self):
        return np.ones(shape=[self.label_dim])
    def zeros(self):
        return np.zeros(shape=[self.label_dim])
    
class BatchedCategoricalLabel(CategoricalLabel):
    def __init__(self, label_dict: List[str] = None):
        super(BatchedCategoricalLabel, self).__init__(label_dict)
    
    def batch_ones(self, batch_size):
        return np.ones(shape=[batch_size,self.label_dim])
    
    def batch_zeros(self, batch_size):
        return np.zeros(shape=[batch_size,self.label_dim])

    def get_batch_label_vectors_by_names(self, names, batch_size):
        labels = self.get_label_vectors_by_category_names(names)
        return np.repeat(labels[:,:,np.newaxis], batch_size, axis=2)
    

class SequenceCategoricalLabel():
    def __init__(self, sequence_length, label_dim, label_dict: Dict[int,str] = None):
        self.sequence_length = sequence_length
        self.label_dim = label_dim
        self.label_dict = label_dict
        self.category_label_generator = CategoricalLabel(label_dim, label_dict)
    
    def get_category_sequence(self, category_int_arr):
        return np.array([self.category_label_generator.get_label_by_id(i) for i in category_int_arr])
    