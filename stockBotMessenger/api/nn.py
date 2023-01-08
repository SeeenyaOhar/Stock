import json
from NNS.inquiryProcessor.BERT.inquiryEstimatorBERT import InquiryAnalyzerBERT, InquiryAnalyzerBERTModel
from NNS.inquiryProcessor.inquiry_analyzer_abs import InquiryAnalyzer # abstraction
from NNS.feedback_classifier.BERT.model import FeedbackClassifierBERT, FeedbackClassifierBERTModel
from NNS.feedback_classifier.feedback_classifier_abc import FeedbackClassifier # abstraction
from NNS.dataset_search.dataset_searcher_abc import DatasetSearcher
import NNS.inquiryProcessor.bert as bert
ia_name = "inquiry_analyzer"
fa_name = "feedback_analyzer"
dss_name = "dataset_searcher"
ia_ds_name = "inquiry_analyzer_ds"
models_p = "stockBotMessenger\\api\\models.json"

with(open(models_p) as f):
    models = json.load(f)
    inquiry_analyzer_p = models[ia_name]
    feedback_analyzer_p = models[fa_name]
    dataset_searcher_p = models[dss_name]
    inquiry_analyzer_ds_p = models[ia_ds_name]

def get_inquiry_analyzer() -> InquiryAnalyzer:
    # model = InquiryAnalyzerBERTModel.from_file(inquiry_analyzer_p)
    encoder, preprocessor = bert.get_bert_details()
    model = InquiryAnalyzerBERTModel.from_file(inquiry_analyzer_p, inquiry_analyzer_ds_p)
    inquiry_analyzer = InquiryAnalyzerBERT(model)
    return inquiry_analyzer

def get_feedback_analyzer() -> FeedbackClassifier:
    feedback_analyzer = FeedbackClassifierBERT()
    return feedback_analyzer
    
def get_dataset_searcher() -> DatasetSearcher:
    dataset_searcher = DatasetSearcher() # this is an abstraction
    return dataset_searcher