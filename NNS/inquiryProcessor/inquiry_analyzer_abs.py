from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class InquiryAnalyzer(ABC):
    @abstractmethod
    def classify(self, a: list or str) -> Tuple[str, np.ndarray]:
        pass
