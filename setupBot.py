# THIS IS A SCRIPT TO CREATE A JSON FILE FOR A BOT

# THE JSON FILE SHOULD CONTAIN:
# 0. an ID of THE BOT
# 1. A NAME THE BOT
# 2. A DESCRIPTION OF THE BOT
# 3. PHONE NUMBERS TO CONTACT THE COMPANY THAT SELLS THE PRODUCTS
# 4. A WEBSITE/SOCIAL NETWORK PROFILE WHERE THE ITEMS ARE SOLD
import os
import phonenumbers
import pymongo

from bot import Bot


class BotCreator:
    """
    BotCreator is used to create a bot and post it to the database, set it up and more.
    """
    def __init__(self, name: str, description: str, phone_number: phonenumbers.PhoneNumber, resource_url: str):
        """
        Creates a BotCreator object.
        :param name: A name of the bot.
        :param description: A description of the bot.
        :param phone_number:  A phone number the user or the administrator of Stock can contact the client(a shop)
        :param resource_url: A link to the client(shop).
        """
        assert (type(name) == str)
        assert (type(description) == str)
        assert (type(phone_number) == phonenumbers.PhoneNumber)
        assert (type(resource_url) == str)

        self.bot = Bot(name, description, phone_number, resource_url)

        # connect a db
        # TODO: USE A MONGO DB
