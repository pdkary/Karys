import numpy as np

class CategoricalLabel():
    def __init__(self,label_dim):
        self.label_dim = label_dim
    
    def get_single_category(self,label_int):
        z = np.zeros(shape=[self.label_dim])
        z[label_int] = 1
        return z
    
    def get_multi_category(self,labels_arr):
        z = np.zeros(shape=[self.label_dim])
        for label_int in labels_arr:
            z[label_int] = 1
        return z
    
    def get_all_categories(self):
        return np.ones(shape=[self.label_dim])
    
    def get_no_categories(self):
        return np.zeros(shape=[self.label_dim])
    
    def succ(self):
        if self.label_dim == 1:
            #single value binary
            return self.get_single_category(0)
        elif self.label_dim == 2:
            #two value binary
            return self.get_single_category(1)
        else:
            #multi category
            return self.get_all_categories()
    
    def fail(self):
        if self.label_dim == 2:
            #two value binary
            return self.get_single_category(0)
        else:
            return self.get_no_categories()

class BatchedCategoricalLabel():
    def __init__(self, label_dim):
        self.label_dim = label_dim
        self.category_label_generator = CategoricalLabel(label_dim)
    
    def get_all_categories(self, batch_size):
        return np.array([self.category_label_generator.get_all_categories() for x in range(batch_size)])
    
    def get_no_categories(self, batch_size):
        return np.array([self.category_label_generator.get_no_categories() for x in range(batch_size)])
    
    def get_multi_categories(self, labels_arr, batch_size):
        return np.array([self.category_label_generator.get_multi_category(labels_arr) for x in range(batch_size)])
    
    def get_single_categories(self, label_int, batch_size):
        return np.array([self.category_label_generator.get_multi_category(label_int) for x in range(batch_size)])
    
    def succ(self, batch_size):
        return np.array([self.category_label_generator.succ() for x in range(batch_size)])

    def fail(self, batch_size):
        return np.array([self.category_label_generator.fail() for x in range(batch_size)])

class SequenceCategoricalLabel():
    def __init__(self, sequence_length, label_dim):
        self.sequence_length = sequence_length
        self.label_dim = label_dim
        self.category_label_generator = CategoricalLabel(label_dim)
    
    def get_category_sequence(self, category_int_arr):
        return np.array([self.category_label_generator.get_single_category(i) for i in category_int_arr])
    