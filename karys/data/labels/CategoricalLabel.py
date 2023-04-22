from typing import List
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

        all_categories = [*categories, *flags]
        self.df = pd.DataFrame(all_categories, columns=["categories"])
        self.one_hot_encoder = OneHotEncoder(sparse=False, dtype=np.float32).fit(self.df)
        self.one_hot_matrix = self.one_hot_encoder.transform(self.df)
    
    @property
    def shape(self):
        return (self.label_dim,)

    def get_label_ids_by_names(self,names):
        vals = self.df.loc[self.df["categories"].isin(names)]
        return [vals.loc[self.df["categories"] == n].index[0] for n in names]
    
    def get_label_vector_by_category_name(self, name):
        return self.get_label_vectors_by_ids(self.get_label_ids_by_names([name]))
    
    def get_label_vectors_by_ids(self, ids):
        return self.one_hot_matrix[ids]
    
    def get_label_vectors_by_category_names(self,names):
        return self.get_label_vectors_by_ids(self.get_label_ids_by_names(names))
    
    def get_label_names_by_ids(self, ids):
        return self.df["categories"][ids]
    