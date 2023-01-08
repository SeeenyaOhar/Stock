import json
import os


class ContractionsJSON:
    path = os.getcwd() + "\\contractions.json"

    def __init__(self):
        self.dictionary = {}

    def add(self, contraction, replacement):
        self.dictionary[contraction] = replacement

    def remove(self, contraction):
        replacement = self.dictionary.pop(contraction)
        return replacement

    def init_dictionary(self):
        try:
            with open(self.path, "rt") as i:
                data = i.read()
                self.dictionary = json.loads(data)

        except IOError as ioe:
            print(str(ioe))

    def save_json(self):
        try:
            with open(self.path, "wt") as i:
                data = json.dumps(self.dictionary)
                i.write(data)
        except IOError as ioe:
            print(str(ioe))
