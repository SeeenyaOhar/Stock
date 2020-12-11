# In order to run here, "en_core_web_md" has to be installed
import spacy
import numpy as np


def cleanUpTheInquiry(inquiry : str):
    removeCharacters = [".", ",", "+", "-", "=", ":", "-", "?", "!"]
    for i in removeCharacters:
        inquiry.replace(i, "")
    return inquiry

class InquiryConverter:
    """
    Converts the inquiry(string) to embeddings from spacy(will be changed to pytorch embeddings soon)
    """
    _dictionary = None

    def __init__(self, inquiry, language="en"):
        if InquiryConverter._dictionary is None:
            InquiryConverter._dictionary = WordsDictionary.getWordsDictionary(language)
        self.data = inquiry
        self.words = inquiry.split(" ")

    def convertTonpArray(self) -> np.array:
        arrays = np.zeros((len(self.words), 300), dtype=np.double)
        i = 0
        for curWord in self.words:
            word = cleanUpTheInquiry(curWord)
            array = np.array(InquiryConverter._dictionary.vocab[word].vector)
            arrays[i] = array.astype(dtype=np.double)
            i += 1
        return arrays



class InquiryArrayConverter:
    """
    Converts an array of inquiries to embeddings(the same as InquiryConverter)
    """
    _dictionary = None

    def __init__(self, inquiries, language="en"):  # We're passing in the numpy array with the inquiries given
        self.inquiries = inquiries
        self.language = language
        if InquiryArrayConverter._dictionary is None:
            InquiryArrayConverter._dictionary = WordsDictionary.getWordsDictionary(self.language)

    def convertToNumpyNumbers(self):
        # So we have an array of inquiries
        # Each one has words
        # And each word has to be converted into numpy numbers
        result = []

        for i in self.inquiries:
            words = i.split(" ")
            wordsNumpyArray = np.zeros((len(words), 300), dtype=np.double)
            i = 0
            for word in words:
                array = np.array(InquiryArrayConverter._dictionary.vocab[word].vector)
                wordsNumpyArray[i] = array.astype(dtype=np.double)
                i += 1
            result.append(wordsNumpyArray)
        return result


# this class has got to return the vectors of all words(currently only English words)

class WordsDictionary:
    """
    Manages dictionaries for creating embeddings from spacy library.
    """
    @staticmethod
    def getWordsDictionary(language="en"):
        packageName = ""
        # English
        if language == "en":
            packageName = "en_core_web_lg"

        if packageName == "":
            raise ValueError("Language as such was not found.")
        else:
            return spacy.load(packageName)
