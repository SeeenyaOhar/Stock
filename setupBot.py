# THIS IS A SCRIPT TO CREATE A JSON FILE FOR A BOT

# THE JSON FILE SHOULD CONTAIN:
# 1. A NAME OF THE BOT
# 2. A DESCRIPTION OF THE BOT
# 3. PHONE NUMBERS TO CONTACT THE COMPANY THAT SELLS THE PRODUCTS
# 4. A WEBSITE/SOCIAL NETWORK PROFILE WHERE THE ITEMS ARE SOLD
import os
import torch
import torch.nn.utils.rnn as utils
import numpy as np
from NNS.inquiryProcessor.inquiryConverter import InquiryConverter, InquiryArrayConverter
from NNS.inquiryProcessor.inquiryEstimator import InquiryAnalyzer
from NNS.inquiryProcessor.dataset import InquiryDataset


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


def train(anal):
    # assigning epochs
    epochs = 1000
    # getting dataset
    np_training_dataset = InquiryDataset.getTrainingDataset()
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
    anal.save("C:\\Users\\senya\\Desktop\\mmm.weights")


# initializing an analyzer
anal = InquiryAnalyzer(True)
anal.load("C:\\Users\\senya\\Desktop\\mmm.weights")
np_training_dataset = InquiryDataset.getTrainingDataset()
# splitting data and converting to a right form
training_dataset = np_training_dataset
training_input = anal.packSequence(
    InquiryArrayConverter(["Hi, how many cores does IPhone's processor have?",
                           "Hey, i need a phone for "
                           "my parent asap. It has to be cheap "
                           "and fast.",
                           "Hi, what's the phone number?", "How much does it cost to deliver to Juliet?",
                           "I need a red dress for a party. Do you have one?",],
                          language="en").convertToNumpyNumbers())
anal.eval()
print(str(anal.forward(training_input, cellStateSize=len(training_input.sorted_indices)).float().round()))
