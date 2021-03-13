# In order to run here, "en_core_web_md" has to be installed
import spacy
import numpy as np
from NNS.inquiryProcessor.contractions import ContractionsJSON


def cleanUpTheInquiry(inquiry: str):
    """Cleans the inqury from some redundant characters(,.!? and so on)

    Args:
        inquiry (str): The inquiry.

    Returns:
        (str): The cleaned inquiry.
    """
    inquiry = inquiry.lower()
    inquiry = cleanCharacters(inquiry)
    inquiry = cleanContractions(inquiry)
    return inquiry


def cleanCharacters(inquiry):
    removeCharacters = [".", ",", "+", "-", "=", ":", "-", "?", "!", '"']
    for i in removeCharacters:
        inquiry = inquiry.replace(i, "")
    return inquiry


def cleanContractions(inquiry):
    contractions = ContractionsJSON()
    contractions.initDictionary()

    for contraction in contractions.dictionary:
        if contraction in inquiry:
            print(contraction)
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
            InquiryConverter._dictionary = WordsDictionary.getWordsDictionary(
                language)
        self.data = inquiry
        self.words = inquiry.split(" ")

    def convertTonpArray(self) -> np.array:
        inq = cleanUpTheInquiry(self.data)
        self.words = inq.split(" ")
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

    # We're passing in the numpy array with the inquiries given
    def __init__(self, inquiries, language="en"):
        self.inquiries = inquiries
        self.language = language
        if InquiryArrayConverter._dictionary is None:
            InquiryArrayConverter._dictionary = WordsDictionary.getWordsDictionary(
                self.language)

    def convertToNumpyNumbers(self):
        # So we have an array of inquiries
        # Each one has words
        # And each word has to be converted into numpy numbers
        result = []

        for i in self.inquiries:
            inq = cleanUpTheInquiry(i)
            words = inq.split(" ")

            wordsNumpyArray = np.zeros((len(words), 300), dtype=np.double)
            i = 0
            for curWord in words:
                word = cleanUpTheInquiry(curWord)
                array = np.array(
                    InquiryArrayConverter._dictionary.vocab[word].vector)
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
