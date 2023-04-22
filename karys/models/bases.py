from abc import ABC
from typing import List, Tuple
from keras.models import Model, Sequential
from keras.layers import Input, BatchNormalization
from keras.utils import plot_model
import numpy as np
from karys.layers.WeightedAdd import WeightedAdd

class GraphableModelBlock(Model, ABC):
    created_graphs = 0
    tabs=0
    def __init__(self, *args, **kwargs):
        super(GraphableModelBlock, self).__init__(*args, **kwargs)
        self.layer_definitions = []
        self.additional_outputs = []
    
    # simple way to call a stack of layers on a tensor
    def call(self, input_tensor, training=True):
        # print(self)
        # print("CALL:")
        # print("\tINPUT: ", input_tensor)
        x = input_tensor
        extra_outs = []
        for layer in self.layer_definitions:
            # print(layer)
            # if isinstance(layer, GraphableModelBlock):
            #     x = layer.call(x, training=training)
            # else:
            x = layer(x, training=training)
            if isinstance(x, tuple):
                x, e_outs = x
                extra_outs.extend(e_outs)
            # print("\tLAYER: ", layer)
            # print("\tOUT: ", x)
        return x, extra_outs 
    
    def get_inputs(self):
        print("IN GET INPUTS: ",self.input_shape)
        if isinstance(self.input_shape, Tuple):
            in_shape = self.input_shape[1:] if self.input_shape[0] is None else self.input_shape
            # print("\tIN SHAPE: ", in_shape)
            return Input(shape=in_shape)
        elif isinstance(self.input_shape, List):
            return [Input(shape=s) for s in self.input_shape]

    # A convenient way to get model summary 
    # and plot in subclassed api
    def build_graph(self, recursive_graph_count = 0):
        self.created_graphs += recursive_graph_count
        inputs = self.get_inputs()
        outputs, extra_outs = self.call(inputs)
        outputs = [outputs, *extra_outs]
        return Model(inputs=inputs, outputs=outputs, name="model_" + str(self.created_graphs))
    
    def plot_graphable_model(self, plot_path, tabs=0):
        # print("Building graphs for: ", type(self))
        self_graph = self.build_graph()
        plot_model(self_graph, plot_path + "/" + self.name + ".png", expand_nested=True, show_shapes=True, show_dtype=True, show_layer_names=True, show_layer_activations=True)
        for layer in self_graph.layers:
            # print("\t"*tabs, layer.__class__, layer.input_shape, layer.input)
            if issubclass(type(layer), GraphableModelBlock):
                # print("Building submodel: ", type(layer))
                layer.plot_graphable_model(plot_path, tabs+1)
    
    @property
    def weighted_add_layers(self):
        weighted_adds = []
        for l in self.layer_definitions:
            if isinstance(l, WeightedAdd):
                weighted_adds.append(l)
            elif isinstance(l, GraphableModelBlock):
                weighted_adds.extend(l.weighted_add_layers)
        return weighted_adds