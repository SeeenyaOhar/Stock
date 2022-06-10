import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from typing import Tuple
import numpy as np
import tensorflow as tf
from inquiryEstimator import InquiryAnalyzerAssistant
import tensorflow_hub as hub
import keras
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer
from dataset import InquiryDataset
import matplotlib.pyplot as plt
from NNS.inquiryProcessor.inquiry_analyzer_abs import InquiryAnalyzer
class InquiryAnalyzerBERTModel(keras.Model):

    def __init__(self, tfhub_handle_preprocess: str, tfhub_handle_encoder: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_layer = keras.Input(shape=(), dtype=tf.string, name="INPUT")
        self.preprocess_layer = hub.KerasLayer(tfhub_handle_preprocess, name="PREPROCESS")
        self.encoder_layer = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name="BERT_ENCODER")
        self.dropout = keras.layers.Dropout(0.1, name="DROPOUT")
        self.dense = keras.layers.Dense(10, activation="softmax", name="FINAL_CLASSIFIER")

    def call(self, inputs, training=None, mask=None):
        # y = self.input_layer(inputs)
        y = self.preprocess_layer(inputs)
        y = self.encoder_layer(y)['pooled_output']
        y = self.dropout(y)
        y = self.dense(y)
        return y
    @staticmethod
    def from_file(filepath, ds_path, epochs=100):
        model = keras.models.load_model(filepath, custom_objects={"AdamWeightDecay": InquiryAnalyzerBERT.get_optimizer(InquiryAnalyzerDatasetManagerBERT.get_ds(32, ds_path), epochs)})
        return model

class InquiryAnalyzerBERT(InquiryAnalyzer):
    """Classifies the Inquiry to 10 different classes depending on the context:
    1. user_interaction_needed = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    2. contact = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    3. dataset_search = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    4. delivery = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    5. order = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    6. welcome = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    7. feedback = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    8. checkout = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
    9. checkoutRequest = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    10. recommendation = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])\n
    For more information: https://github.com/SeeenyaOhar/Stock
    """
    def __init__(self, model):
        self.model = model

    def train(self, ds: tf.data.Dataset, epochs: int=100, callback: keras.callbacks.ModelCheckpoint=None):
        optimizer = InquiryAnalyzerBERT.get_optimizer(ds, epochs)
        loss = InquiryAnalyzerBERT.get_loss()
        metrics = InquiryAnalyzerBERT.get_metrics()
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        if callback == None:
            logs = self.model.fit(x=ds, epochs=epochs)
        else:
            logs = self.model.fit(x=ds, epochs=epochs, callback=callback)
        return logs
    
    
    def classify(self, a: list or str) -> Tuple[str, np.ndarray]:
        """Classifies the inquiry to 10 different classes that are described in InquiryDataset class(dataset.py)

        Args:
            a (list): Inquiry Set

        Returns:
            Tuple[str, np.ndarray]: Classified
        """
        if type(a) is str:
            a = [a]
        for i in a:
            assert type(i) == str
        result_tensor = self.model.call(tf.convert_to_tensor(tf.convert_to_tensor(a)))
        result_np = result_tensor.numpy()
        return InquiryAnalyzerAssistant.classifierstringar(result_np.round()), result_np.round()
    
    @staticmethod
    def get_optimizer(ds: tf.data.Dataset, epochs: int):
        steps_per_epoch = tf.data.experimental.cardinality(ds).numpy()
        num_train_steps = steps_per_epoch * epochs
        # used to increase the learning rate of the first 10% of the dataset
        num_warmup_steps = int(0.1 * num_train_steps)

        init_lr = 3e-5
        optimizer = optimization.create_optimizer(init_lr=init_lr, 
                                                  num_train_steps=num_train_steps,
                                                  num_warmup_steps=num_warmup_steps, 
                                                  optimizer_type="adamw")
        return optimizer
    
    @staticmethod
    def get_loss():
        loss = tf.keras.losses.BinaryCrossentropy()
        return loss
    @staticmethod
    def get_metrics():
        metrics = tf.metrics.BinaryAccuracy()
        return metrics
    
class InquiryAnalyzerDatasetManagerBERT:
    @staticmethod
    def get_ds(BATCH_SIZE, dataset_path: str) -> tf.data.Dataset:
        npdataset = InquiryDataset.get_training_dataset(dataset_path)
        train_examples = tf.convert_to_tensor(npdataset[:, 0])
        train_labels = np.stack(npdataset[:, 1])
        assert train_labels.shape[1] == 10
        train_labels = train_labels.reshape((npdataset[:, 1].shape[0], 10))  # tensorflow can't convert labels easily
        train_labels = tf.convert_to_tensor(train_labels)
        dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
        train_ds = dataset.batch(BATCH_SIZE, drop_remainder=False).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        return train_ds


if __name__ == '__main__':
    # write your test code here
    print(InquiryAnalyzerBERTModel.get_bert_details())

