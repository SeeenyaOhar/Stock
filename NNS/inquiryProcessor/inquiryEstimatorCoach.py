import os
import torch
import torch.nn.utils.rnn as utils
import numpy as np
from NNS.inquiryProcessor.inquiryConverter import InquiryConverter, InquiryArrayConverter
from NNS.inquiryProcessor.inquiryEstimator import InquiryAnalyzerRNN
from NNS.inquiryProcessor.dataset import InquiryDataset
ANAL_WEIGHTS_FILEPATH = "C:\\Users\\User\\mmm.weights"

def labelsNumpy(array):
    """
    Converts a list to a numpy array with labels.
    :param array: A list to be converted.
    :return: A numpy array with labels.
    """
    result = np.zeros((len(array), 10))
    for n, i in enumerate(array):
        nparray = np.array(i)
        result[n] = nparray
    return result


def train(anal, epochs=1000):
    # assigning epochs
    anal.load(ANAL_WEIGHTS_FILEPATH)
    # getting dataset
    np_training_dataset = InquiryDataset.get_training_dataset()
    # splitting data and converting to a right form
    training_dataset = np_training_dataset
    training_input = anal.packSequence(InquiryArrayConverter(np_training_dataset[:, 0],
                                                             language="en").convertToNumpyNumbers())
    # getting the first axis which is the input
    np_labels = labelsNumpy(np_training_dataset[:, 1])
    # getting the second axis(the labels for the input given)
    training_labels = torch.from_numpy(np_labels)
    # training data
    result = anal.trainData(training_input, training_labels, epochs)
    anal.save(ANAL_WEIGHTS_FILEPATH)


def test(anal):
    anal.load(ANAL_WEIGHTS_FILEPATH)
    np_training_dataset = InquiryDataset.get_training_dataset()
    # splitting data and converting to a right form
    training_dataset = np_training_dataset
    v = np_training_dataset[0:3, 0].tolist()
    v2 = np_training_dataset[-3:-1, 0].tolist()
    sample = v + v2 + [np_training_dataset[-1, 0]]
    y = np_training_dataset[0:3, 1].tolist() + np_training_dataset[-3:-1, 1].tolist() + [np_training_dataset[-1, 1]]
    training_input = anal.packSequence(
        InquiryArrayConverter(sample,
                              language="en").convertToNumpyNumbers())
    anal.eval()
    result = str(anal.forward(training_input, cellStateSize=len(
        training_input.sorted_indices)).round())

    print("Sample: {0} \n Y: {1}\nResult: {2}".format(sample, y, result))

if __name__ == '__main__':
    # initializing an analyzer
    anal = InquiryAnalyzerRNN(True)

    # torch.device("cuda")
    train(anal, epochs=1000)
    # train(anal, epochs=100)