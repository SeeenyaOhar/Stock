import torch
import numpy as np
from NNS.inquiryProcessor.inquiryEstimator import InquiryAnalyzer
from NNS.inquiryProcessor.inquiryConverter import InquiryArrayConverter


def convert(inq):
    converter = InquiryArrayConverter(inq)
    return converter.convertToNumpyNumbers()


# initializing an analyzer
anal = InquiryAnalyzer(False)
anal.load("C:\\Users\\senya\\Desktop\\mmm.weights")

test = ["Hi, what's your phone number?"]
testV = convert(test)
testTensor = anal.packSequence(testV)

result = anal.forward(testTensor)
print(result)
