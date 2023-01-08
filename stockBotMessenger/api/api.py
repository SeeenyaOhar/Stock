from flask import *
import os
import sys
from requests import head
currentdir = os.path.dirname(__file__)
stock_dir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(os.path.join(stock_dir))
from nn import get_inquiry_analyzer, get_feedback_analyzer, get_dataset_searcher
stock = Flask(__name__)


@stock.route("/class_message", methods= ['POST'])
def class_message():
    """
    Classifies the message based on the message included in the request.
    Returns an array pointing at what class the message is.
    E.g.
    "[0 0 0 1 0 0 0 0 0 0]"
    """
    if request.method == "POST":
        if not request.is_json:
            abort(500)
        data = request.get_json()
        message = data["message"]
        result = inquiry_analyzer.classify(message)
        for i in result[0]:
            out_message += i
        response = make_response(out_message, 200)
        response.mimetype = "text/plain"
        return response
        
def class_feedback():
    """
    Classifies the feedback message.
    Returns a floating decimal point number.
    E.g.
    "0.65", "0.54", "0.832134"
    """
    pass

def search_item():
    pass

if __name__ == "__main__":
    # it depends on these objects
    # inquiry analyzer
    # feedback analyzer
    # dataset searcher
    inquiry_analyzer = get_inquiry_analyzer()
    stock.run()