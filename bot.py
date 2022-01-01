import phonenumbers


class Bot:
    def __init__(self, name: str, description: str, phone_number: phonenumbers.PhoneNumber, resource_url: str):
        """
        Creates a bot object over some of its characteristics.
        :param name: str : A name of the bot.
        :param description: A description of the bot.
        :param phone_number: A phone number the user or the administrator of Stock can contact the client(a shop)
        :param resource_url: A link to the client(shop).
        """
        assert (type(phone_number) == phonenumbers.PhoneNumber)
        self.name = name
        self.description = description
        self.phone_number = phone_number
        self.resource_url = resource_url

    def __str__(self):
        string = "Bot: [\n" \
                 "name: '{name}'," \
                 "description: '{description}', \n" \
                 "phone_number: '{phone_number}', \n" \
                 "resource_url: '{resource_url}' \n" \
                 "]".format(name=self.name, description=self.description, phone_number=self.phone_number,
                            resource_url=self.resource_url)
        return string
