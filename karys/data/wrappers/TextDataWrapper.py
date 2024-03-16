import re
import string
import tensorflow as tf

def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  return tf.strings.regex_replace(lowercase,
                                  '[%s]' % re.escape(string.punctuation), '')
class TextDataWrapper():
    def __init__(self, 
                 input_text_path: str,
                 input_length: int,
                 output_length: int,
                 train_test_ratio: float = 0.7):
        self.validation_percentage = 1 - train_test_ratio
        self.input_path = input_text_path
        with open(self.input_path, 'r') as f:
            self.input_text = f.read()
        
        self.words = self.input_text.split()
        self.vocab = sorted(set(self.words))
        self.vocab_size = len(self.vocab)
        self.word_to_index = {word: index for index, word in enumerate(self.vocab)}
        self.index_to_word = {index: word for index, word in enumerate(self.vocab)}
        
        self.input_sequences = []
        self.output_sequences = []
        for i in range(len(self.words) - input_length - output_length):
            input_seq = self.words[i:i + input_length]
            output_seq = self.words[i + input_length:i + input_length + output_length]
            self.input_sequences.append([self.word_to_index[word] for word in input_seq])
            self.output_sequences.append([self.word_to_index[word] for word in output_seq])

    @property
    def size(self):
        return len(self.input_sequences)
    
    def get_train_dataset(self):
        validation_size = int(self.validation_percentage*self.size)
        sentence_inputs = self.input_sequences[:-validation_size]
        sentence_outputs = self.output_sequences[:-validation_size]
        train_data = list(zip(sentence_inputs,sentence_outputs))
        return train_data
    
    def get_validation_dataset(self):
        validation_size = int(self.validation_percentage*self.size)
        sentence_inputs = self.input_sequences[-validation_size:]
        sentence_outputs = self.output_sequences[-validation_size:]
        test_data = list(zip(sentence_inputs, sentence_outputs))
        return test_data

    def translate_sentence(self, sentence):
        return " ".join([self.index_to_word[x] for x in sentence if x in self.index_to_word])
    
    def translate_sentences(self,sentence_list):
        return [self.translate_sentence(sentence) for sentence in sentence_list]

    def show_sentence_n(self,n):
        sin = self.input_sequences[n]
        son = self.output_sequences[n]
        print("sentence in:\n", self.translate_sentence(sin), sin.shape)
        print("\nsentence out:\n", self.translate_sentence(son), son.shape)
