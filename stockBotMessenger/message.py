import numpy as np
from bot_configuration import BotConfiguration


class MessageHelper:
    def __init__(self, bot: BotConfiguration):
        self.bot = bot

    def __alert_admin(self):
        # TODO: Implement alert method
        pass

    def answer(self, message: str, classification: np.ndarray) -> str:
        """
        Depending on the classification(list of 10 numbers either 0 or 1) returns an appropriate message to the user.
        :param message: str - message that the bot answers to.
        :param classification: np.ndarray - prediction of the AI.
        :return: str - message.
        """

        if classification[0,4] == 1:
            # ORDER
            print("ORDER - PROCESSING")
        if classification[0,2] == 1:
            print("SEARCH PROCESSING")
            # TODO: Integrate a search AI machine
        if classification[0,3] == 1:
            print("DELIVERY")
            return self.bot.DELIVERY_MESSAGE
        if classification[0,7] == 1:
            print("CHECKOUT")
            return self.bot.CHECKOUT_MESSAGE
        if classification[0,0] == 1:
            print("USER INTERACTION NEEDED")
            self.__alert_admin()  # TODO: Implement this method and connect directly to the admin.
            return self.bot.USER_INTERACTION_MESSAGE
        if classification[0,1] == 1:
            print("CONTACT")
            return self.bot.CONTACT_MESSAGE
        if classification[0,8] == 1:
            print("CHECKOUT_REQUEST")
            return self.bot.CHECKOUT_REQUEST_MESSAGE
        if classification[0,6] == 1:
            print("FEEDBACK")
            return self.bot.FEEDBACK_MESSAGE
        if classification[0,5] == 1:
            print("WELCOME")
            return self.bot.WELCOME_MESSAGE
        if classification[0,9] == 1:
            print("RECOMMENDATION")
            return self.bot.RECOMMENDATION_MESSAGE
