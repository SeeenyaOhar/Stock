from flask import Flask
stock = Flask(__name__)


# TODO: Insert the @stock.route("{route}") here for the bot
def process_message():
    messageKindEstimator = InquiryEstimator()



if __name__ == '__main__':
    stock.run()