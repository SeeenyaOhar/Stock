import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod
class FeedbackClassifier(ABC):
    @abstractmethod
    def classify(self, a: list or str):
        pass