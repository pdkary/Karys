from keras.layers import Layer

class OutputLayer():
    def __init__(self, layer: Layer) -> None:
        self.layer = layer
