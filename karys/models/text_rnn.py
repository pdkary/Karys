from keras.layers import Embedding, LSTM, Dense, Reshape, Softmax
from karys.models.bases import GraphableModelBlock

class TextRNN(GraphableModelBlock):
    def __init__(self, vocab_size: int, vector_size: int, input_length: int, output_length: int, rnn_units: int):
        super(GraphableModelBlock, self).__init__()
        self.input_length = input_length

        self.layer_definitions = [
            Embedding(vocab_size, vector_size, input_length=input_length),
            LSTM(rnn_units),
            Dense(output_length * vocab_size, activation='relu'),
            Reshape((output_length, vocab_size)),
            Softmax(axis=-1),
        ]

    @property
    def input_shape(self):
        return (self.input_length,)
