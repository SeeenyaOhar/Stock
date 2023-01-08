from re import L
import tensorflow as tf
import keras
import tensorflow_hub as hub
from official.nlp import optimization
from NNS.feedback_classifier.feedback_classifier_abc import FeedbackClassifier
from NNS.inquiryProcessor.BERT.inquiryEstimatorBERT import InquiryAnalyzerBERT
from typing import Tuple
import numpy as np
class FeedbackClassifierBERTModel(keras.Model):
    def __init__(self, preprocessor: str, encoder: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_layer = keras.layers.Input(shape=(None,), dtype=tf.string)
        self.preprocessor = hub.KerasLayer(preprocessor, name="PREPROCESSING_LAYER")
        self.encoder = hub.KerasLayer(encoder, name="ENCODER_LAYER")
        self.dropout = keras.layers.Dropout(0.1, name="DROPOUT")
        self.dense = keras.layers.Dense(1, activation="softmax", name="CLASSIFIER_DENSE_LAYER")
    def call(self, x):
        y = self.preprocessor(x)
        y = self.encoder(x)['pooled_output']
        y = self.dropout(y)
        y = self.dense(y)
        return y

class FeedbackClassifierBERT(FeedbackClassifier):
    """
    Classifies the string of feedback either to positive or negative one.
    """
    def __init__(self, model):
        self.model = model
    @staticmethod
    def get_optimizer(ds: tf.data.Dataset, epochs: int=100, lr: float=3e-5):
        steps_per_epoch = tf.data.experimental.cardinality(ds).numpy()
        num_train_steps = steps_per_epoch * epochs
        # 10% of all num_train_steps
        num_warmup_steps = int(0.1 * num_train_steps)
        optimizer = optimization.create_optimizer(init_lr=lr, 
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
    @staticmethod
    def from_file(filepath, ds: tf.data.Dataset, epochs: int=100):
        model = keras.models.load_model(filepath, custom_objects={"AdamWeightDecay": FeedbackClassifierBERT.get_optimizer(ds, epochs)})
        return model
    
    def train(self, 
              ds: tf.data.Dataset,
              epochs: int=100):
        optimizer = FeedbackClassifierBERT.get_optimizer(ds, epochs)
        loss = FeedbackClassifierBERT.get_loss()
        metrics = FeedbackClassifierBERT.get_metrics()
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        logs = self.model.fit(x=ds, epochs=epochs)
        return logs

    def classify(self, a: list or str) -> Tuple[str, np.ndarray]:
        """Classifies the inquiry to 1 classes(1 feedback is positive, 0 it's negative)

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
        result_str = ""

        for i in result_np:
            result_str += "POSITIVE " if i[0].round() == 1 else "NEGATIVE"
            
        return result_str, result_np

        
    