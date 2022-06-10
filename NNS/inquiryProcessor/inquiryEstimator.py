import numpy as np
from abc import ABC
from typing import Tuple
class ClassificationConverter(ABC):
    def convert(self, a: np.ndarray) -> str:
        pass
class StringClassificationConverter():
    def convert(self,a: np.ndarray) -> str:
        if a[4] == 1:
            result = "ORDER"
        if a[2] == 1:
            result = "SEARCH"
        if a[3] == 1:
            result = "DELIVERY"
        if a[7] == 1:
            result = "CHECKOUT"
        if a[0] == 1:
            result = "USER INTERACTION NEEDED"
        if a[1] == 1:
            result = "CONTACT"
        if a[8] == 1:
            result = "REQUEST"
        if a[6] == 1:
            result = "FEEDBACK"
        if a[5] == 1:
            result = "WELCOME"
        if a[9] == 1:
            result = "RECOMMENDATION"
        return result
class NumberClassificationConverter:
    def convert(self, a: np.ndarray) -> str:
        result = 0
        for i in range(a.shape[0]):
            if a[i] == 1:
                result = i
                break
        return str(result)
class InquiryAnalyzerAssistant:
    
    @staticmethod
    def classifierstring(a: np.ndarray, cc=NumberClassificationConverter()):
        return cc.convert(a)
    @staticmethod
    def classifierstringar(a: np.ndarray, cc=NumberClassificationConverter()):
        """Returns a string representation of the (1, 10) array according to these labels:
        1. user_interaction_needed = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        2. contact = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        3. dataset_search = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        4. delivery = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        5. order = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        6. welcome = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        7. feedback = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        8. checkout = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        9. checkoutRequest = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        10. recommendation = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])\n
        For more information: https://github.com/SeeenyaOhar/Stock

        Args:
            a (np.ndarray): An array of labels(shape(1,10))

        Returns:
            str: A string representation of a classification.
        """
        result = ["" for i in a]
        for i, el in enumerate(a):
            result[i] = cc.convert(el)
        return result