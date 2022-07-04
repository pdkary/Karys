import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from data.configs.TextDataConfig import TextDataConfig
from data.wrappers.DataWrapper import DataWrapper

class TextDataWrapper(DataWrapper):
    def __init__(self, 
                 word_index, 
                 input_sentences,
                 label_sentences,
                 data_config: TextDataConfig, 
                 train_test_ratio: float = 0.7,):
        super(TextDataWrapper, self).__init__(data_config, train_test_ratio)
        self.word_index = word_index
        self.input_sentences = input_sentences
        self.label_sentences = label_sentences
        self.index_to_word = {v:k for k, v in word_index.items()}

    @property
    def size(self):
        return len(self.input_sentences)
    
    @property
    def word_size(self):
        return len(self.word_index)
    
    @classmethod
    def load_from_file(cls, filename: str, data_config: TextDataConfig):
        with open(filename, 'r') as f:
            corpus = f.readlines()
        tokenizer = Tokenizer(num_words=data_config.vocab_size)
        tokenizer.fit_on_texts(corpus)

        word_index = tokenizer.word_index
        train_sequences = tokenizer.texts_to_sequences(corpus)

        input_sets = []
        label_sets = []
        IL= data_config.input_length
        OL = data_config.output_length
        L = IL+OL
        for sequence in train_sequences:
            N = len(sequence)
            for i in range(N-L):
                input_sets.append(sequence[i:i+IL])
                label_sets.append(sequence[i+IL:i+L])
        return cls(word_index, np.array(input_sets), np.array(label_sets), data_config)
    
    def get_train_dataset(self):
        train_size = int(self.train_test_ratio*self.size)
        sentence_inputs = self.input_sentences[:train_size]
        sentence_labels = self.label_sentences[:train_size]
        train_data = tf.data.Dataset.from_tensor_slices((sentence_inputs,sentence_labels))
        return train_data
    
    def get_test_dataset(self):
        train_size = int(self.train_test_ratio*self.size)
        sentence_inputs = self.input_sentences[train_size:]
        sentence_labels = self.label_sentences[train_size:]
        test_data = tf.data.Dataset.from_tensor_slices((sentence_inputs,sentence_labels))
        return test_data

    def translate_sentence(self, word_code_list, scale=False):
        return " ".join([self.index_to_word[x] for x in word_code_list if x > 0 ])
    
    def translate_sentences(self,sentence_list):
        return [self.translate_sentence(sentence) for sentence in sentence_list]

    def show_sentence_n(self,n):
        sin = self.input_sentences[n]
        son = self.label_sentences[n]
        print("sentence in:\n", self.translate_sentence(sin), sin.shape)
        print("\nsentence out:\n", self.translate_sentence(son), son.shape)
