# Logic
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.rnn as utils
import numpy as np
import time
# UI

import curses as curses
import matplotlib.pyplot as plt


class InquiryAnalyzer(nn.Module):
    """
    Basically, neural network that classifies the inquiry embeddings based on the context.
    """

    # Initializing model here
    def __init__(self, terminalUI=False):
        # init the super of pytorch neural network
        super(InquiryAnalyzer, self).__init__()
        # Shrinking down the data
        # Or in other words cutting down the data we don't need
        if terminalUI:
            self.epochScr = curses.initscr()
        self.l1 = torch.nn.Linear(300, 100)
        self.l12 = torch.nn.Linear(100, 50)
        self.lstm = torch.nn.LSTM(50, 25, 3)
        self.l2 = torch.nn.Linear(25, 20)
        self.softmax = torch.nn.Linear(20, 10)
        self.float()

    def packSequence(self, sequence):
        """
            Pads and packs sequence.
                    :param sequence: A sequence of list type
        Args:
            sequence (list): A sequence.

        Returns:
            [PackPaddedSequence]: Packed padded sequence.
        """
        a = self.setupSequence(sequence)
        lengths = [len(i) for i in a]

        b = utils.pad_sequence(a, batch_first=True)

        return utils.pack_padded_sequence(b, batch_first=True, lengths=lengths, enforce_sorted=False).float()

    def setupSequence(self, sequence):
        """
        Processes sequence to an appropriate form for the neural network.
        :param sequence: A sequence of list type.
        :return: A processed sequence.
        """
        result = sequence.copy()
        for n, i in enumerate(sequence):
            result[n] = torch.from_numpy(sequence[n])
        return result

    def forward(self, x, cellStateSize=1):
        """
        Forward propogation.
        :param x: Tensor of words [1, 300]
        :return: Tensor [?, 1, 10] with the inquiry classifications.
        """
        assert (isinstance(x, torch.nn.utils.rnn.PackedSequence))

        # x here is the inquiry
        # check the correctness of the size
        cell_state = torch.zeros(3, cellStateSize, 25)
        # check the correctness of the size
        hidden_state = torch.zeros(3, cellStateSize, 25)
        # Checking if all of the elements of array x(which is an inquiry basically)
        result = None
        # going through every word

        if x.data[0].shape[0] != 300:
            raise ValueError(
                "The size of x has to be 300(vector features of the word)")
        else:
            currentInput = x
            # (1) Densed layer(shrinking down)
            currentInput = PackedSequenceHelper.squash_packed(
                currentInput, self.l1)
            currentInput = PackedSequenceLeakyReluHelper.squash_packed_relu(currentInput, )
            # currentInput = PackedSequenceHelper.squash_packed(currentInput)
            # (2) Densed layer(shrinking down even more)
            currentInput = PackedSequenceHelper.squash_packed(
                currentInput, self.l12)
            # currentInput = PackedSequenceHelper.squash_packed(currentInput)
            currentInput = PackedSequenceLeakyReluHelper.squash_packed_relu(currentInput)
            # (3) LSTM Layer - based on the hidden state and cell state we predict what does the sentence mean
            # In other words, what kind of inquiry user has made


            (out, h0) = self.lstm(currentInput, (hidden_state, cell_state))
            hidden_state = h0
            currentInput = out

            currentInput = PackedSequenceHelper.squash_packed(
                currentInput, self.l2)
            # result = PackedSequenceHelper.squash_packed(currentInput)
            result = PackedSequenceLeakyReluHelper.squash_packed_relu(currentInput)


        # (4) - Softmax layer(output layer) | Classifying the inquiry
        result = PackedSequenceHelper.squash_packed(result, self.softmax)
        result = PackedSequenceHelper.squash_packed_softmax(result, dim=1)
        return self._sequenceLabels(result)

    def trainData(self, x, y, epochs=1000):
        """
        Trains the data over a sequence. Uses the MSE Loss.
        :param x: Input data.
        :param y: Labels used to estimate the loss over the dataset.
        :param epochs: A number of epochs.
        :return: List of loss values
        """
        assert (isinstance(x, torch.nn.utils.rnn.PackedSequence))
        assert (isinstance(y, torch.Tensor))
        self.train()
        # assert(isinstance(y, torch.Tensor))
        # assert(isinstance(epochs, int))
        # optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        optimizer = optim.Adam(self.parameters())
        error = torch.nn.MSELoss()
        losses = []
        for i in range(epochs):
            self._changeEpoch(i)
            # batchSize = 0
            # dataBatchStartIndex = 0
            # for n in x.batch_sizes:
            #     batchSize = n
            #     dataBatchEnd = dataBatchStartIndex + batchSize
            #     currentBatch = x.data[dataBatchStartIndex:dataBatchEnd]
            output = self.forward(
                x.float(), cellStateSize=len(x.sorted_indices))
            loss_value = error(output, y.float())
            self._changeAccuracy(loss_value)
            loss_value.backward()
            optimizer.step()
            losses.append(loss_value)

        optimizer.zero_grad()
        output = self.forward(x.float(), cellStateSize=len(x.sorted_indices))
        loss_value = error(output, y.float())
        self._changeAccuracy(loss_value)
        loss_value.backward()
        optimizer.step()
        self.epochScr.erase()
        print("Final result: {0}".format(output.float().round()))
        print("\nLoss: {0}".format(loss_value.float()))
        self.epochScr.refresh()
        plt.plot(losses)
        plt.ylabel("Losses")
        plt.xlabel("Epoch")
        plt.title("Losses over {0} epochs".format(epochs))
        plt.show()

    def _sequenceLabels(self, packed):
        """
        Returns a Tensor of labels from a packed sequence.
        :param packed:
        :return:
        """
        temp = utils.pad_packed_sequence(packed, batch_first=True)
        padded = temp[0]
        lengths = temp[1]
        result = torch.zeros(len(padded), 10)
        for n, i in enumerate(padded):
            curEl = padded[n, lengths[n] - 1]
            result[n] = curEl
        return result

    def _changeEpoch(self, epoch):
        """
        Changes the epoch number to {epoch} in Terminal. Uses curses library.
        :param epoch: Epoch number that the Terminal text has to changed to.
        :return: None
        """
        self.epochScr.erase()
        self.epochScr.addstr("Epoch: {0} \nAccuracy: ".format(epoch))
        self.epochScr.refresh()
        time.sleep(0.001)

    def _changeAccuracy(self, accuracy):
        """
        Changes the accuracy to {accuracy} in Terminal. Uses curses library.
        :param accuracy: Accuracy number.
        :return: None
        """
        self.epochScr.addstr("{0}".format(accuracy))
        self.epochScr.refresh()
        time.sleep(0.001)

    def save(self, path):
        """
        Saves the models dictionary(state_dict) to the path file.
        """
        try:
            torch.save(self.state_dict(), path)
            print("The model has been saved to {0}".format(path))
        except Exception as e:
            print(str(e))

    def load(self, path):
        """Loads the weights and biases for the model from path file.

        Args:
            path (str): The path of the file that contains the weights and biases.
        """
        try:
            self.load_state_dict(torch.load(path))
            print("The model has been loaded with {0}".format(path))
        except Exception as e:
            print(str(e))


class PackedSequenceHelper:
    @staticmethod
    def squash_packed(x, fn=torch.tanh):
        """
        Computes fn with an argument x.data, where x is PackedSequence.
        :param x: PackedSequence that the function is processed with respect to its data.
        :param fn: Function which we process.
        :return: PackedSequence with the processed output of function fn with respect to x.data.
        """
        return torch.nn.utils.rnn.PackedSequence(fn(x.data), x.batch_sizes,
                                                 x.sorted_indices, x.unsorted_indices)

    @staticmethod
    def squash_packed_softmax(x, dim=2):
        """
        Computes softmax with an argument x.data where x is PackedSequence
        :param x: PackedSequence that the softmax is processed with respect to its data
        :param dim: A dimension along which the softmax will be computed.
        :return: A processed PackedSequence with softmax.
        """
        return torch.nn.utils.rnn.PackedSequence(F.softmax(x.data, dim), x.batch_sizes,
                                                 x.sorted_indices, x.unsorted_indices)


class PackedSequenceLeakyReluHelper:
    @staticmethod
    def squash_packed_relu(x, slope=0.7):
        """
        Squashes the PackedSequence with Leaky Relu Function.
        :param x: The PackedSequence whose data we process with Leaky Relu.
        :param slope: The slope number for softmax.
        :return: Squashed PackedSequence.
        """
        return torch.nn.utils.rnn.PackedSequence(F.leaky_relu(x.data, negative_slope=slope), x.batch_sizes,
                                                 x.sorted_indices, x.unsorted_indices)
