import os
import json

bert_model_name = 'small_bert/bert_en_uncased_L-4_H-512_A-8'


def set_bert_name(name):
    global bert_model_name
    bert_model_name = name


def get_bert_details():
    """
    Returns handle encoder and bert model links.
    :return: tfhub_handle_preprocess: str, tfhub_handle_encoder: str
    """
    try:
        file_path = os.path.dirname(os.path.realpath(__file__)) + "/bert_models.json"
        with(open(file_path, "r") as f):
            bert_models = json.load(f)
            map_name_to_handle = bert_models["encoder"]
            map_model_to_preprocess = bert_models["preprocessor"]

            tfhub_handle_encoder = map_name_to_handle[bert_model_name]
            tfhub_handle_preprocess = map_model_to_preprocess[bert_model_name]

            print(f'BERT model selected           : {tfhub_handle_encoder}')
            print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')
            return tfhub_handle_preprocess, tfhub_handle_encoder
    except FileNotFoundError as err:
        print("File was not found: bert_models.json;\n" +
              "Cannot read the models")
