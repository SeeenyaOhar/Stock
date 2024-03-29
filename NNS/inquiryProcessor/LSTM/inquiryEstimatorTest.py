import torch
import numpy as np
from NNS.inquiryProcessor.inquiryEstimator import InquiryAnalyzerRNN
from NNS.inquiryProcessor.inquiryConverter import InquiryArrayConverter
ANAL_WEIGHTS_FILEPATH = "C:\\Users\\User\\mmm.weights"


def convert(inq):
    converter = InquiryArrayConverter(inq)
    return converter.convertToNumpyNumbers()

if __name__ == "__main__":
    # initializing an analyzer
    anal = InquiryAnalyzerRNN(False)
    anal.load(ANAL_WEIGHTS_FILEPATH)

    test = ["Hi, what's your phone number?"]
    testV = convert(test)
    testTensor = anal.packSequence(testV)

    result = anal.forward(testTensor)
    print(result)
