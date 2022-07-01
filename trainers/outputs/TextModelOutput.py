
import json
import numpy as np


class TextModelOutput(object):
    def __init__(self):
        self.output_sets = {}
    
    def add_output(self, input_sentences, output_sentences):
        input_key = " | ".join(input_sentences)
        self.output_sets[input_key] = output_sentences
    
    def save_to_file(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.output_sets, f)