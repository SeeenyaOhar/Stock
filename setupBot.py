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
from NNS.inquiryProcessor.inquiryEstimator import InquiryAnalyzer, InquiryDataset


def labelsNumpy(array):
    result = np.zeros((len(array), 10))
    for n, i in enumerate(array):
        nparray = np.array(i)
        result[n] = nparray
    return result


def packSequence(sequence):
    a = setupSequence(sequence)
    lengths = [len(i) for i in a]

    b = utils.pad_sequence(a, batch_first=True)

    return utils.pack_padded_sequence(b, batch_first=True, lengths=lengths, enforce_sorted=False).double()


def setupSequence(sequence):
    result = sequence.copy()
    for n, i in enumerate(sequence):
        result[n] = torch.from_numpy(sequence[n])
    return result


inquiry = InquiryConverter("Hello World").convertTonpArray()
inquiry = torch.from_numpy(inquiry).type(torch.double)
anal = InquiryAnalyzer().double()
epochs = 1000
np_training_dataset = InquiryDataset.getTrainingDataset()
training_dataset = np_training_dataset
training_input = packSequence(InquiryArrayConverter(np_training_dataset[:, 0],
                                                    language="en").convertToNumpyNumbers())  # getting the first axis which is the input
np_labels = labelsNumpy(np_training_dataset[:, 1])
training_labels = torch.from_numpy(np_labels).double()  # getting the second axis(the labels for the input given)

result = anal.trainData(training_input, training_labels)
print(result)