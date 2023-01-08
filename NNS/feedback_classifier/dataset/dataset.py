import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__)) # NNS\feedback_classifier\dataset
parentdir = os.path.dirname(currentdir) # NNS\feedback_classifier
targetdir = os.path.dirname(parentdir) # NNS\
sys.path.append(targetdir)
from services import pandas as pds
import tensorflow as tf
import pandas as pd
import numpy as np
from inquiryProcessor.dataset import InquiryDataset
class FeedbackDataset:
    @staticmethod
    def get_ds(filepath: str):
        """Returns the feedback dataset. 

        Args:
            filepath (str): A path to the dataset._
        """
        print("Reading content from {0}".format(filepath))
        dataset = pd.read_csv(filepath, warn_bad_lines=True)
        dataset = dataset.to_numpy()
        dataset = dataset[:, [1,2]]
        classes = dataset[:, 1]
        for i, el in enumerate(classes):
            classes[i] = np.array(el)
        return dataset
        
    @staticmethod
    def import_ds(ds: np.ndarray, path: str):
        """Imports the data from dataset and saves to the file with the path specified.

        Args:
            ds (tf.Data.Dataset): The dataset that will be imported.
            path (str): The path where the dataset will be saved.
        

        Returns:
            np.ndarray: A feedback dataset.
        """
        # Just for those who are curious
        # This function will be usually used for importing the data from inquiries dataset
        # We will just filter it out for classes, then select only feedback class inquiries
        # and import this data to the file
        
        # let's import the inquiries dataset first
        feedback_dataset = []
        for i in ds:
            if i[1][6] == 1:
                feedback_dataset.append(i)
        feedback_dataset = np.array(feedback_dataset)[:, [0]]
        pd.DataFrame(feedback_dataset).to_csv(path)
        return feedback_dataset

if __name__ == "__main__":
    inquiries_path = "D:\\documents\\code\\Stock\\NNS\\inquiryProcessor\\inquiries_dataset.csv"
    save_path = "D:\\documents\\code\\Stock\\NNS\\feedback_classifier\\dataset\\unlabeled_feedback.csv"
    ds = FeedbackDataset.import_ds(
        ds=InquiryDataset.get_training_dataset(inquiries_path),
        path=save_path)