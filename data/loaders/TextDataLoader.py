import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from data.configs.TextDataConfig import TextDataConfig

def load_from_text(corpus, text_data_config: TextDataConfig):
    # Tokenize our training data
    tokenizer = Tokenizer(num_words=text_data_config.num_words)
    tokenizer.fit_on_texts(corpus)

    # Get our training data word index
    word_index = tokenizer.word_index

    # Encode training data sentences into sequences
    train_sequences = tokenizer.texts_to_sequences(corpus)
    valid_input_sequences = [x for x in train_sequences if len(x) <= text_data_config.sentence_length]
    input_padded = pad_sequences(valid_input_sequences, padding='post', truncating='post', maxlen=text_data_config.sentence_length)

    input_sets = []
    predicted_label_sets = []
    i_offset = text_data_config.input_sentences
    p_offset = text_data_config.predicted_sentences
    offset = i_offset + p_offset
    for i in range(len(input_padded) - offset):
        input_sets.append(input_padded[i:i+i_offset])
        predicted_label_sets.append(input_padded[i+i_offset:i+offset])
    return word_index, np.array(input_sets), np.array(predicted_label_sets)

def load_from_file(filename, text_data_config: TextDataConfig):
    with open(filename, 'r') as f:
        corpus = f.readlines()
    return load_from_text(corpus, text_data_config)
