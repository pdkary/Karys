from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from karys.data.wrappers.TextDataWrapper import TextDataWrapper
from karys.models.text_rnn import TextRNN
from karys.trainers.TextRNNTrainer import TextRNNTrainer

INPUT_LENGTH = 16
OUTPUT_LENGTH = 4
VECTOR_SIZE = 128
RNN_UNITS = 64

def load_models():
    global INPUT_LENGTH, OUTPUT_LENGTH, VECTOR_SIZE
    data_loader = TextDataWrapper("./examples/text_rnn/test_input/corpus.txt", INPUT_LENGTH, OUTPUT_LENGTH)
    vocab_size = data_loader.vocab_size
    rnn_model = TextRNN(vocab_size, VECTOR_SIZE,  INPUT_LENGTH, OUTPUT_LENGTH, RNN_UNITS)
    rnn_model.plot_graphable_model('./examples/text_rnn/test_output/architecture_diagrams')
    rnn_model = rnn_model.build_graph()
    rnn_model.summary()

    optimizer = Adam(learning_rate=0.001)
    loss = CategoricalCrossentropy(label_smoothing=0.1)

    trainer = TextRNNTrainer(rnn_model, optimizer, loss, data_loader)
    return rnn_model, trainer

def train(batch_size, num_batches, epochs):
    model, trainer = load_models()
    for i in range(epochs):
        loss = trainer.train(batch_size, num_batches)
        print(f"EPOCH {i}: {loss}")
        if i % 50 == 0:
            print(trainer.print_most_recent_output())
    trainer.save('./examples/text_rnn/test_output/model')

