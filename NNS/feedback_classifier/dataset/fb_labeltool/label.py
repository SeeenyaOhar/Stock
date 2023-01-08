"""
This is a tool to label the feedback dataset and save it.
"""
import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from dataset import FeedbackDataset
import pandas as pd
import numpy as np
def difference(a: np.ndarray, b: np.ndarray):
    c = []
    for el in a:
        for el2 in b:
            if el != el2:
                c.append(el)
    return c
def main():
    print("\nHey, how are you doing my friend?\nReady for some labels?")
    print("I will get the dataset for you. But type the path to the unlabeled first.")
    unlabeled_feedback_path = input()
    # GET THE UNLABELED DATASET
    unl_ds = pd.read_csv(unlabeled_feedback_path).to_numpy()
    unl_ds = unl_ds[:, 1]
    # GET THE LABELED DATASET
    print("Now the labeled one. ")
    labeled_feedback_path = input()
    l_ds = pd.read_csv(labeled_feedback_path).to_numpy()[:, 1]
    # FIND THE DIFFERENCE
    ds = difference(unl_ds, l_ds)
    # LABEL
    print("Alright, I got the dataset. We are good to go. Here is the first feedback sample")
    feedback = ds[0]
    print(feedback)
    print("\nWhat you think about it? Print 1 if it's positive and 0 if it's not.")
    label = int(input())
    labels = [[feedback, label]]
    print("Alright, got the first one. We're good! Here is the next one!\n"+ 
          "If you don't want to label the feedback just press enter!")
    for i in ds[1:]: # some problem here fix it
        feedback = i
        print(feedback + "\n")
        label = input()
        if (label == ""):
            break
        labels.append([feedback, label])
    # MERGE LABELS
    labels = np.array(labels)
    l_ds = np.concatenate(labels, l_ds)
    # SAVE
    save_path = os.path.join(labeled_feedback_path)
    print("Alright, good job sir! Have a nice day! I will save the dataset for you in \n", save_path)
    pd.DataFrame(l_ds).to_csv(save_path)
if __name__ == "__main__":
    main()