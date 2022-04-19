import phonenumbers
from message import MessageHelper
from bot_configuration import BotConfiguration, MessagePack
import numpy
import requests
import sys
sys.path.append("D:\\Documents\\Code\\Stock")
from NNS.inquiryProcessor.inquiryEstimator import InquiryAnalyzer
from NNS.inquiryProcessor.BERT.inquiryEstimatorBERT import InquiryAnalyzerBERT, InquiryAnalyzerBERTModel, \
    InquiryAnalyzerDatasetManagerBERT
from flask import Flask, request
from typing import Tuple
import os



stock = Flask(__name__)


def get_message_pack(phone_number: phonenumbers.PhoneNumber, email: str) -> list:
    user_interaction_needed = "Hey, we're going to process your message soon and text you back. Thank you for waiting."
    contact = "Hi, we work 9 to 5 and you can contact us here: \n{0}\n{1}".format(
        str(phone_number.country_code) + str(phone_number.national_number), email)
    dataset_search = "Hi, hang on, we'll answer your question soon."
    delivery = "Hi, we deliver goods to every country in the world. Delivery fee is 2$. Have a nice day!"
    order = "Hi, we will process your order soon. Have a nice day!"
    welcome = "Hi, we are a goods company and differ from others with products of awesome quality. How may I help you?"
    # TODO: Classify feedback to either good or bad
    feedback = "Thanks for your feedback."
    checkout = "We offer these ways to pay: PayPal, MasterCard, robocassa. What would you like to choose?"
    checkout_request = "Alright, good, let me see if the money has been acquired"
    recommendation = "Hey, here is the thing you were looking for: "
    return [user_interaction_needed, contact, dataset_search, delivery, order, welcome, feedback, checkout,
            checkout_request, recommendation]


telegram_bot_info = {
    "token": "5277508536:AAGDFgz6X7zropxTDL-8AmaK7Vmk1ambIF4", "name": "stock_store_bot"}
request_url = "https://api.telegram.org/bot{0}/{1}"

phone_number = phonenumbers.parse("+380987669293")
email = "arseniyohar@gmail.com"
message_pack = MessagePack(get_message_pack(phone_number, email))
bot_configuration = BotConfiguration(
    name="Stock Test", phone_number=phone_number, email=email, message_pack=message_pack, description="This is a test "
                                                                                                      "description")


def get_analyzer(type_of_analyzer: str):
    """Returns a message analyzer that can classify messages depending on the context.

    Returns:
        _type_: _description_
    """
    if type_of_analyzer == 'bert':
        print("LOADING BERT CLASSIFICATION MODEL...")
        try:
            
            bert_model = InquiryAnalyzerBERT.get_model_from_file(str(sys.argv[1]),
                                                                InquiryAnalyzerDatasetManagerBERT.get_ds(
                                                                    64, sys.argv[2]),
                                                                epochs=1000)
            print("BERT MODEL HAS BEEN LOADED SUCCESSFULLY")
            message_classifier = InquiryAnalyzerBERT(bert_model)
            return message_classifier
        
        except IndexError as e:
            print(e, "\nBERT Model is invalid. Try correcting it and try later.")
            os.abort()
    else:
        return None


def send_classification(classification: Tuple[str, numpy.ndarray], chat_id, user_message):
    # TODO: add a bot configuration to the backend
    helper = MessageHelper(bot_configuration)  # <------------------
    message = helper.answer(user_message, classification[1])
    url = request_url.format(telegram_bot_info["token"], "sendMessage")
    data = {"chat_id": chat_id, "text": message}
    requests.post(url, data=data)


@stock.route("/", methods=["GET", "POST"])
def process_message():
    """
    Route "/". Processes the message that the bot receives.
    :return:
    """
    if request.method == "POST":
        message: str = request.json["message"]["text"]
        print(message)
        if message[0] == "/":
            if message == "/start" or message == "/help":
                url = request_url.format(
                    telegram_bot_info["token"], "sendMessage")
                requests.post(url, data={
                    "text": "Welcome to the club buddy!\n"
                            "Here you can write a message that you would text to the store and we will "
                            "classify it to different categories.",
                    "chat_id": request.json["message"]["chat"]["id"]})
        else:
            result = analyzer.classify([message])
            print(result)
            send_classification(
                result, request.json["message"]["chat"]["id"], message)
    return {"ok": True}


if __name__ == '__main__':
    analyzer = get_analyzer("bert")
    print(analyzer)
    stock.run()
