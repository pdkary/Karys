import tensorflow as tf
from data.configs.TextDataConfig import TextDataConfig
from data.loaders import TextDataLoader
from data.wrappers.DataWrapper import DataWrapper


class TextDataWrapper(DataWrapper):
    def __init__(self, 
                 word_index, 
                 input_sentences, 
                 predicted_sentences,
                 data_config: TextDataConfig, 
                 train_test_ratio: float = 0.7,):
        super(TextDataWrapper, self).__init__(data_config, train_test_ratio)
        self.word_index = word_index
        self.index_to_word = {v:k for k, v in word_index.items()}
        self.input_sentences = input_sentences
        self.predicted_sentences = predicted_sentences

    @property
    def size(self):
        return len(self.input_sentences)

    @classmethod
    def load_from_file(cls, filename: str, data_config: TextDataConfig):
        word_index, input_sentences, output_sentences =  TextDataLoader.load_from_file(filename, data_config)
        return cls(word_index, input_sentences, output_sentences, data_config)
    
    def get_train_dataset(self):
        train_size = int(self.train_test_ratio*self.size)
        if self.data_config.ignore_output:
            train_data = tf.data.Dataset.from_tensor_slices((self.input_sentences[:train_size]))
        else:
            train_data = tf.data.Dataset.from_tensor_slices((self.input_sentences[:train_size], self.predicted_sentences[:train_size]))
        return train_data
    
    def get_test_dataset(self):
        train_size = int(self.train_test_ratio*self.size)
        if self.data_config.ignore_output:
            test_data = tf.data.Dataset.from_tensor_slices((self.input_sentences[train_size:]))
        else:
            test_data = tf.data.Dataset.from_tensor_slices((self.input_sentences[train_size:], self.predicted_sentences[train_size:]))
        return test_data

    def is_valid_code(self,key):
        return int(key) in self.index_to_word

    def translate_sentence(self, word_code_list):
        return " ".join([self.index_to_word[int(x)] for x in word_code_list if self.is_valid_code(x)])

    

