import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod
class DatasetSearcher(ABC):
    @abstractmethod
    def search(self, a: list or str):
        pass