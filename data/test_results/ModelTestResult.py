from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class ModelTestResult(ABC):
    loss: np.float32
    inputs: List[List[np.float32]]
    labels: List[List[np.float32]]
    preds: List[List[np.float32]]

    @property
    def size(self):
        return len(self.inputs)
