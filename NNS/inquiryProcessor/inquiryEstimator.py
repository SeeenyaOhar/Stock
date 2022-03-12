import numpy as np
from typing import Tuple
class InquiryAnalyzer:
    def classify(self, inquiries: list) -> Tuple[str, np.ndarray]:
        pass
class InquiryAnalyzerAssistant:
    @staticmethod
    def classifierstring(a: np.ndarray):
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
            if a[i, 4] == 1:
                result[i] += "ORDER "
            if a[i, 2] == 1:
                result[i] += "SEARCH "
            if a[i, 3] == 1:
                result[i] += "DELIVERY "
            if a[i, 7] == 1:
                result[i] += "CHECKOUT "
            if a[i, 0] == 1:
                result[i] += "USER INTERACTION NEEDED"
            if a[i, 1] == 1:
                result[i] += "CONTACT"
            if a[i, 8] == 1:
                result[i] += "REQUEST "
            if a[i, 6] == 1:
                result[i] += "FEEDBACK "
            if a[i, 5] == 1:
                result[i] += "WELCOME "
            if a[i, 9] == 1:
                result[i] += "RECOMMENDATION "
        return result