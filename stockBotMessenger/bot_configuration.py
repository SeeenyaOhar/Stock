import phonenumbers
import json


# TODO: Integrate JSON with the type definitions

class MessagePack:

    def __init__(self, message_pack: list):
        """
        :param message_pack: Typical messages of type string should be placed in a list according to this plan:
                             1. USER_INTERACTION
                             2. CONTACT
                             3. DATASET_SEARCH
                             4. DELIVERY
                             5. ORDER
                             6. WELCOME
                             7. FEEDBACK
                             8. CHECKOUT
                             9. CHECKOUT_REQUEST
                             10. RECOMMENDATION
                                     :return:
        """
        self.MESSAGE_PACK = message_pack
        self.USER_INTERACTION = message_pack[0]
        self.CONTACT = message_pack[1]
        self.DATASET_SEARCH = message_pack[2]
        self.DELIVERY = message_pack[3]
        self.ORDER = message_pack[4]
        self.WELCOME = message_pack[5]
        self.FEEDBACK = message_pack[6]
        self.CHECKOUT = message_pack[7]
        self.CHECKOUT_REQUEST = message_pack[8]
        self.RECOMMENDATION = message_pack[9]


class BotConfiguration:
    def __init__(self, name: str, phone_number: phonenumbers.PhoneNumber, email: str, description: str,
                 message_pack: MessagePack):
        self.RECOMMENDATION_MESSAGE = message_pack.RECOMMENDATION
        self.WELCOME_MESSAGE = message_pack.WELCOME
        self.FEEDBACK_MESSAGE = message_pack.FEEDBACK
        self.CHECKOUT_REQUEST_MESSAGE = message_pack.CHECKOUT_REQUEST
        self.CONTACT_MESSAGE = message_pack.CONTACT
        self.USER_INTERACTION_MESSAGE = message_pack.USER_INTERACTION
        self.CHECKOUT_MESSAGE = message_pack.CHECKOUT
        self.DELIVERY_MESSAGE = message_pack.DELIVERY
        self.name = name
        self.phone_number = phone_number
        self.email = email
        self.description = description
