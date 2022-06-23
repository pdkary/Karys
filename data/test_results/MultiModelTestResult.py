from abc import ABC
from dataclasses import dataclass
from typing import Dict

import numpy as np
from data.test_results.ModelTestResult import ModelTestResult


@dataclass
class MultiModelTestResult(object):
    test_results: Dict[str,ModelTestResult]

    @property
    def losses(self) -> Dict[str,np.float32]:
        return {k:v.loss for k,v in self.test_results.items()}
    
    @property
    def garbage_results(self) -> Dict[str,bool]:
        return {k:v.is_result_garbage() for k,v in self.test_results.items()}
