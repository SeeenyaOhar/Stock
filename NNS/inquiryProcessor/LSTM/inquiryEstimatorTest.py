import torch
import numpy as np
from NNS.inquiryProcessor.inquiry_estimator import InquiryAnalyzerRNN
from NNS.inquiryProcessor.inquiry_converter import InquiryArrayConverter
ANAL_WEIGHTS_FILEPATH = "C:\\Users\\User\\mmm.weights"


def convert(inq):
    converter = InquiryArrayConverter(inq)
    return converter.convert_to_npn()

if __name__ == "__main__":
    # initializing an analyzer
    anal = InquiryAnalyzerRNN(False)
    anal.load(ANAL_WEIGHTS_FILEPATH)

    test = ["Hi, what's your phone number?"]
    testV = convert(test)
    testTensor = anal.packSequence(testV)

    result = anal.forward(testTensor)
    print(result)
