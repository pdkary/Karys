from keras.layers import Layer

class MultiInputLayer():
    def __init__(self, layer: Layer, secondary_input: Layer) -> None:
        self.layer = layer
        self.secondary_input = secondary_input