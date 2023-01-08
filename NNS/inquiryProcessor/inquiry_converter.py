# In order to run here, "en_core_web_md" has to be installed
import spacy
import numpy as np
from NNS.inquiryProcessor.contractions import ContractionsJSON


def clean_up(inquiry: str):
    """Cleans the inquiry from some redundant characters(,.!? and so on)

    Args:
        inquiry (str): The inquiry.

    Returns:
        (str): The cleaned inquiry.
    """
    inquiry = inquiry.lower()
    inquiry = clean_chars(inquiry)
    inquiry = clean_contractions(inquiry)
    return inquiry


def clean_chars(inquiry):
    """
    Removes special symbols that are not valuable to the spacy embeddings.
    :param inquiry:
    :return:
    """
    remove_characters = [".", ",", "+", "-", "=", ":", "-", "?", "!", '"']
    for i in remove_characters:
        inquiry = inquiry.replace(i, "")
    return inquiry


def clean_contractions(inquiry):
    """
    Cleans up the contractions("what's up", "rofl")
    :param inquiry:
    :return:
    """
    contractions = ContractionsJSON()
    contractions.init_dictionary()

    for contraction in contractions.dictionary:
        if contraction in inquiry:
            replacement = contractions.dictionary[contraction]
            inquiry = inquiry.replace(contraction, replacement)

    return inquiry


class InquiryConverter:
    """
    Converts the inquiry(string) to embeddings from spacy(will be changed to pytorch embeddings soon)
    """
    _dictionary = None

    def __init__(self, inquiry, language="en"):
        if InquiryConverter._dictionary is None:
            InquiryConverter._dictionary = WordsDictionary.get_words_dictionary(
                language)
        self.data = inquiry
        self.words = inquiry.split(" ")

    def convert_to_np(self) -> np.array:
        inq = clean_up(self.data)
        self.words = inq.split(" ")
        arrays = np.zeros((len(self.words), 300), dtype=np.double)
        i = 0
        for curWord in self.words:
            word = clean_up(curWord)
            array = np.array(InquiryConverter._dictionary.vocab[word].vector)
            arrays[i] = array.astype(dtype=np.double)
            i += 1
        return arrays


class InquiryArrayConverter:
    """
    Converts an array of inquiries to embeddings(the same as InquiryConverter)
    """
    _dictionary = None

    # We're passing in the numpy array with the inquiries given
    def __init__(self, inquiries, language="en"):
        self.inquiries = inquiries
        self.language = language
        if InquiryArrayConverter._dictionary is None:
            InquiryArrayConverter._dictionary = WordsDictionary.get_words_dictionary(
                self.language)

    def convert_to_npn(self):
        # So we have an array of inquiries
        # Each one has words
        # And each word has to be converted into numpy numbers
        result = []

        for i in self.inquiries:
            inq = clean_up(i)
            words = inq.split(" ")

            words_numpy_array = np.zeros((len(words), 300), dtype=np.double)
            i = 0
            for curWord in words:
                word = clean_up(curWord)
                array = np.array(
                    InquiryArrayConverter._dictionary.vocab[word].vector)
                words_numpy_array[i] = array.astype(dtype=np.double)
                i += 1
            result.append(words_numpy_array)
        return result


class WordsDictionary:
    """
    Manages dictionaries for creating embeddings from spacy library.
    """

    @staticmethod
    def get_words_dictionary(language="en"):
        package_name = ""
        # English
        if language == "en":
            package_name = "en_core_web_lg"  # we use a large package for the most precision

        if package_name == "":
            raise ValueError("Language as such was not found.")
        else:
            return spacy.load(package_name)
