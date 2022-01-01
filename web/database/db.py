import pymongo as pm
from bot import Bot
class BotDB:
    def __init__(self):
        self.bots_collection = BotDB.get_database()
    @staticmethod
    def get_database():
        """
        Returns a bots mongo db database collection("mongodb://127.0.0.1:27017/")['Bots']
        :return: Collection : a mongo db database collection
        """
        client = pm.MongoClient("mongodb://127.0.0.1:27017/")
        bots_collection = client["stock"]["Bots"]
        return bots_collection

    def add(self, bot: Bot):
        pass

    def remove(self):
        pass


