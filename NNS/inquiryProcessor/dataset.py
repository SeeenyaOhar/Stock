import numpy as np
import csv as csv
import pandas as pd


class InquiryDataset:
    """
    Manages the training datasets.
    """
    dataset_path = "inquiries_dataset.csv"
    user_interaction_needed = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    contact = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    dataset_search = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    delivery = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    order = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    welcome = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    feedback = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    checkout = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
    checkoutRequest = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    recommendation = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    @staticmethod
    def get_training_dataset() -> np.ndarray:
        """
        Reads the numpy training dataset from csv file(located in dataset_path in InquiryDataset class).
        :return:
        """
        df = pd.read_csv(InquiryDataset.dataset_path)
        dataset = df.to_numpy()
        dataset = dataset[:, [1, 2]] # in the file we have also ids in zero column, so let's select only labels
        # and inquiries
        classes = dataset[:, 1]
        for i, el in enumerate(classes):
            classes[i] = InquiryDataset.pandas_csv_real_to_string(el)
        return dataset

    @staticmethod
    def pandas_csv_real_to_string(a: str) -> np.ndarray:
        """
        Pandas numpy array is a string. To convert to numpy array we use this method.
        Converts '[1 0 1 0 0 0 0 0 0]' to 'array([[1, 0, 1, 0, 0, 0, 0, 0, 0]]'
        :param a: csv string
        :return: np.ndarray
        """

        b = a.split('[')[1].split(']')[0].split(' ')
        print(b)
        c = np.zeros([1, len(b)])
        print(c)
        for i, el in enumerate(b):
            print(i, " ", el)
            c[0, i] = int(el)
        return c

    @staticmethod
    def get_temp_dataset() -> np.ndarray:
        """
                Returns a training dataset [*, 2] that contains inquiries and classifications respectively.
                :return: Returns a numpy array [*, 2] with inquiries and classifications
                """
        return np.array([["Hi, what's your phone number?", InquiryDataset.contact],
                         ["Hi, may I get the phone number?",
                          InquiryDataset.contact],
                         ["Hi, I want to order this product",
                          (InquiryDataset.order + InquiryDataset.dataset_search)],
                         ["Good morning, I'm interested in this product. What's the price of this one?",
                          InquiryDataset.dataset_search],
                         ["Are there any of this product in stock?",
                          InquiryDataset.dataset_search],
                         ["What are the ways to deliver the product?",
                          InquiryDataset.delivery],
                         ["How long does the delivery take?",
                          InquiryDataset.delivery],
                         ["Are any of this product available?",
                          InquiryDataset.dataset_search],
                         ["What about the delivery?", InquiryDataset.delivery],
                         ["Does it have this characteristic?",
                          InquiryDataset.user_interaction_needed],
                         ["We want to cooperate with your company.",
                          InquiryDataset.user_interaction_needed],
                         ["We would be glad to work with your company indeed.",
                          InquiryDataset.user_interaction_needed],
                         ["Hi, name, is that you?",
                          InquiryDataset.user_interaction_needed],
                         ["Is the delivery available to Ukraine?",
                          InquiryDataset.delivery],
                         ["Is the delivery available to this country?",
                          InquiryDataset.delivery],
                         ["What's the price of the delivery?",
                          InquiryDataset.delivery],
                         ["May I pay for the this product when I receive it?",
                          InquiryDataset.delivery],
                         ["What is included in the set?",
                          InquiryDataset.dataset_search],
                         ["What is included with the product?",
                          InquiryDataset.dataset_search],
                         ["Is the delivery free?", InquiryDataset.delivery],
                         ["How long is it going to take to get it to my city?",
                          InquiryDataset.delivery],
                         ["Hey, what's the price of the new Nike?",
                          InquiryDataset.dataset_search],
                         ["Hi, are any of those shoes available?",
                          InquiryDataset.dataset_search],
                         ["Good morning, I'm looking for a new phone for my kid. It shouldn't be expensive. Thank you!",
                          InquiryDataset.recommendation],
                         ["Hey, I'm interested in Adidas leggings? How much do you want for them?",
                          InquiryDataset.dataset_search],
                         ["Hey, is the discount still available for the new Puma Cali?",
                          InquiryDataset.dataset_search],
                         ["Hi", InquiryDataset.welcome],
                         ["Hey, do you still have any of those Nike Air's?",
                          InquiryDataset.dataset_search],
                         ["Hi, what's the company that delivers your products?",
                          InquiryDataset.delivery],
                         ["Are those Nike shoes good for winter",
                          InquiryDataset.dataset_search],
                         ["Have you got some budget smartphone? In range of 200-500 $",
                          InquiryDataset.recommendation],
                         ["Are there any problems with delivery because of coronavirus? Isn't it going to take more "
                          "time than usual?", InquiryDataset.delivery],
                         ["Do you have a warranty for the new Nike sneakers?",
                          InquiryDataset.dataset_search],
                         ["When am I going to get the package?",
                          InquiryDataset.delivery],
                         ["Are the new IPhones 12 in stock?",
                          InquiryDataset.dataset_search],
                         ["Why does it take so long to get the package to my house?????",
                          InquiryDataset.delivery],
                         ["Hey, do you have something for a young girl. I need a gift for her birthday!",
                          InquiryDataset.dataset_search],
                         ["How much do you want for the refurbished iphone 6?",
                          InquiryDataset.dataset_search],
                         ["Hey, can you get us one pineapple and a chicago to 2411 Howard Street?",
                          InquiryDataset.order],
                         ["Hi man, how much time to get it to 2411 Howard Street?",
                          InquiryDataset.delivery],
                         ["Is that discount for pineapple pizza still available?",
                          InquiryDataset.user_interaction_needed],
                         ["Hey, do you work today?", InquiryDataset.welcome],
                         ["Hey, do you ship the packages to Europe?",
                          InquiryDataset.delivery],
                         ["Do you ship the packages to America?",
                          InquiryDataset.delivery],
                         ["How many cores does the new iphone have?",
                          InquiryDataset.dataset_search],
                         ["I need a phone for my Mom. Not extremely expensive but not a brick please.",
                          InquiryDataset.recommendation],
                         ["Hi, is there any possibility to cooperate with your company?",
                          InquiryDataset.user_interaction_needed],
                         ["Hi, have you got any vacancies there? I would like to apply for a product manager",
                          InquiryDataset.user_interaction_needed],
                         ["Hey, we need one Chicago pizza to 2111 Howard Street.",
                          InquiryDataset.order],
                         ["I need a pair of Nike Air. ", InquiryDataset.order],
                         ["Hi, those Nike Air are so good man. Thank you all!!!! I love em",
                          InquiryDataset.feedback],
                         ["Hey, thank you for the new item. It's lit!!!",
                          InquiryDataset.feedback],
                         ["Thank you for the pizza! The quality is above the sky!",
                          InquiryDataset.feedback],
                         ["The pizza is lit! Thank a lot!",
                          InquiryDataset.feedback],
                         ["Why do you deliver them in plastic packaging! It's bad for the environment! Replace those "
                          "with organic stuff!", InquiryDataset.feedback],
                         ["The pizza stinks as shit! What the hell is this????",
                          InquiryDataset.feedback],
                         ["Why the hell did you put the tomatoes inside!? I told you not to put them!",
                          InquiryDataset.feedback],
                         ["I've already taken them from the post office! They're so soft and sweet! Thank you and "
                          "love you!", InquiryDataset.feedback],
                         ["I like everything except the price. Could be a bit cheaper. Everything else is great!",
                          InquiryDataset.feedback],
                         ["The delivery took a bit long but nothing bad happened then. The shoes are great. Pull them "
                          "on every day, no problem.", InquiryDataset.feedback],
                         ["No comment! They're amazing!", InquiryDataset.feedback],
                         ["Jesus, thank you! So awesome!", InquiryDataset.feedback],
                         ["Oh, God, thanks a lot! Ur awesome guys, thank you!!!",
                          InquiryDataset.feedback],
                         ["Oh yeah, everything is great. Everything is awesome!",
                          InquiryDataset.feedback],
                         ["Do u have any custom pizza?  I would like to make one!",
                          InquiryDataset.user_interaction_needed],
                         ["Have you got an IPhone 12 in stock?",
                          InquiryDataset.dataset_search],
                         ["Is there Harry Potter in stock?",
                          InquiryDataset.dataset_search],
                         ["Hey, I wanna order Lord Of The Rings please.",
                          InquiryDataset.order + InquiryDataset.dataset_search],
                         [
                             "Hey, how much does the delivery cost for 100 km from your warehouse? And can I take it on my own?",
                             InquiryDataset.delivery],
                         ["Good morning, do ship packages to the South America (Brazilia)?", InquiryDataset.delivery],
                         ["Hey, is Xioami a Chinese phone?",
                          InquiryDataset.dataset_search],
                         ["What's the price of the new Xiaomi Redmi Note 8 Pro?",
                          InquiryDataset.dataset_search],
                         ["Does the new Xiaomi Mi 10 have a Face-ID recognition?",
                          InquiryDataset.dataset_search],
                         ["What's included with One Plus 8 pro?",
                          InquiryDataset.dataset_search],
                         ["What's the price of AirPods Pro?",
                          InquiryDataset.dataset_search],
                         ["Are there any restrictions with USPS?",
                          InquiryDataset.delivery],
                         ["Do you ship the packages throughout the world?",
                          InquiryDataset.delivery],
                         ["What's your phone number?", InquiryDataset.contact],
                         ["I'll phone you when I get the package",
                          InquiryDataset.contact],
                         ["Are you shipping to Russia?", InquiryDataset.delivery],
                         ["What is the company does deliver the packages?",
                          InquiryDataset.delivery],
                         ["What about the delivery?", InquiryDataset.delivery],
                         ["May I get your number please?", InquiryDataset.contact],
                         ["How can I contact you?", InquiryDataset.contact],
                         ["Hi, I want to order Don Quixote, please? How much for that one?",
                          InquiryDataset.order + InquiryDataset.dataset_search],
                         ["Whassup, I wanna order Nike Air Versitile. How much?",
                          InquiryDataset.order + InquiryDataset.dataset_search],
                         ["Have you got this product?",
                          InquiryDataset.dataset_search],
                         ["Hey, may I phone you, please? What's your phone number?",
                          InquiryDataset.contact],
                         [
                             "Hi, I'm looking for a phone for my son. Do you have a cheap one and so my son can play some games? ",
                             InquiryDataset.recommendation],
                         ["Hello, do you ship the packages to China?",
                          InquiryDataset.delivery],
                         ["The phone itself is amazing, but the delivery was really long."
                          " Expected it to come on Monday, but got it only on Friday.", InquiryDataset.feedback],
                         ["Why the hell can't you answer me? I've been waiting for the answer for ten days long!",
                          InquiryDataset.feedback],
                         ["The book is awesome. Guys, I recommend this one.",
                          InquiryDataset.feedback],
                         ["Hey, how you doin? What's your phone number? I would like to ask you about the new air max",
                          InquiryDataset.contact],
                         ["Hi, I really really need the new IPhone for my wife. It's an emergency",
                          InquiryDataset.recommendation],
                         ["Hi, I would like to buy the new Iphone 12, please. Cash, 51 Queen Street"
                          "HEREFORD"
                          "HR7 2FZ", InquiryDataset.checkout],
                         ["Hi, can I pay with cash when I get the package?",
                          InquiryDataset.checkoutRequest],
                         ["Hey, do I have to pay with credit card only?",
                          InquiryDataset.checkoutRequest],
                         ["Hi, do you accept Master Cards?",
                          InquiryDataset.checkout],
                         ["Hello, do I have to prepay for something when ordering a book?",
                          InquiryDataset.checkoutRequest],
                         ["What're the ways to pay for the package?",
                          InquiryDataset.checkoutRequest],
                         ["Hi, I want to buy the new Corsar fireworks? How much do they cost?",
                          InquiryDataset.dataset_search],
                         ["Hi, how much for the Christmas mug? ",
                          InquiryDataset.dataset_search],
                         ["Whassup, how much for the new Wilson Evolution?",
                          InquiryDataset.dataset_search],
                         ["Hey, what ball would you recommend for a complete beginner in basketball?",
                          InquiryDataset.recommendation],
                         ["Hi, what's the price of the blue Chanel shirt?",
                          InquiryDataset.dataset_search],
                         ["Hey, what's the fiber of the Armani shirt?",
                          InquiryDataset.dataset_search],
                         ["Hi, what are the Calvin Klein panties made of?",
                          InquiryDataset.dataset_search],
                         ["Hi, I need something for a party. What have you got?",
                          InquiryDataset.recommendation],
                         ["Hello guys, how long does it take to deliver a shirt to London?",
                          InquiryDataset.delivery],
                         ["Thanks a lot. Got a package and everything is great, the quality's above the sky!",
                          InquiryDataset.feedback],
                         [
                             "Everythign is awesome! The book itself is interesting and breathtaking. The delivery didn't take long. 5 stars",
                             InquiryDataset.feedback],
                         ["Thanks a lot. I'm over the moon!",
                          InquiryDataset.feedback],
                         ["Is the new Wilson Evolution made of the rubber or leather?",
                          InquiryDataset.dataset_search],
                         ["Is the new Wilson Evolution worth it to buy for a beginner or I should get something else?",
                          InquiryDataset.recommendation],
                         ["Is it OK to get a leather ball instead of rubber for a beginner?",
                          InquiryDataset.recommendation],
                         ["May I pay with cash when I get the ball?",
                          InquiryDataset.checkoutRequest],
                         ["What are the paynments methods?",
                          InquiryDataset.checkoutRequest],
                         ["May I pay with a credit card?",
                          InquiryDataset.checkoutRequest],
                         ["May I pay with a Visa Card?",
                          InquiryDataset.checkoutRequest],
                         ["Hi, may I call you up?", InquiryDataset.contact],
                         ["Is it free to call you?", InquiryDataset.contact],
                         ["What's the shop's email address?",
                          InquiryDataset.contact],
                         ["Hi, have you got the new Stephen King book 'It'?",
                          InquiryDataset.dataset_search],
                         ["Hello", InquiryDataset.welcome],
                         ["Hi", InquiryDataset.welcome],
                         ["What's up?", InquiryDataset.welcome],
                         ["Hey", InquiryDataset.welcome],
                         ["Hi, what's the best book for a kid of this year?",
                          InquiryDataset.recommendation],
                         ["Hey, I'm looking for a job.",
                          InquiryDataset.user_interaction_needed],
                         ["Hi, have you got Stehen King The Institute in stock?",
                          InquiryDataset.dataset_search],
                         ["Hey, one Peperoni and Chicago, please",
                          InquiryDataset.dataset_search + InquiryDataset.order],
                         ["Whassup, five California and three spicy tuna role.",
                          InquiryDataset.dataset_search + InquiryDataset.order],
                         ["Hi, I need your phone number", InquiryDataset.contact],
                         ["How you doin' guys? One California and Detroit please.",
                          InquiryDataset.dataset_search + InquiryDataset.order],
                         ["Hi, may I see the menu, please?",
                          InquiryDataset.welcome],
                         ["Hi guys, I'm looking for a job and I've got a diploma of a cook.",
                          InquiryDataset.user_interaction_needed],
                         ["Guys, you suck!", InquiryDataset.user_interaction_needed],
                         ["Do you accept Maestro Credit Cards?",
                          InquiryDataset.checkoutRequest],
                         ["May I get your phone number please?",
                          InquiryDataset.contact],
                         ["Hello, I wanna get the Naruto pens. How much for them?",
                          InquiryDataset.dataset_search],
                         ["What's the price of the new Moleskine Black Edition?",
                          InquiryDataset.dataset_search],
                         ["Hi, do you have something for a musician?",
                          InquiryDataset.recommendation],
                         ["Hi, do you have M size for Versace shirt?",
                          InquiryDataset.dataset_search],
                         ["Hey, what size is available for Armani shirt?",
                          InquiryDataset.dataset_search],
                         ["Hi, what Armani clothes have u got really cheap?",
                          InquiryDataset.recommendation],
                         ["Hi, I need something for my girlfrind. She's already got IPhone 12. What about Watch?",
                          InquiryDataset.recommendation],
                         ["What's the price of delivery to Europe?",
                          InquiryDataset.delivery],
                         ["Hi, have you got any manager vacancy?",
                          InquiryDataset.user_interaction_needed],
                         [
                             "The taste of California is mindblowing! I don't know what have you got there, but that sauce is awesome!",
                             InquiryDataset.feedback],
                         [
                             "Why the hell did you put this vanilla sauce in this pizza? I didn't ask for that. I want my money back now!",
                             InquiryDataset.feedback],
                         ["Have you got Razer Deathadder in stock?",
                          InquiryDataset.dataset_search],
                         [
                             "Hi, I need a budget mouse for cs go. It doesn't have to freeze and break down in the middle of the game. The price range is 30-50$",
                             InquiryDataset.recommendation],
                         ["Should I buy a good membrane keyboard or cheap mechanical one?",
                          InquiryDataset.recommendation],
                         ["I'm still waiting for the delivery. How long do I have to wait",
                          InquiryDataset.delivery],
                         ["Does the delivery usually take a lot of time?",
                          InquiryDataset.delivery],
                         ["Have many Putin card covers have you got, guys?",
                          InquiryDataset.dataset_search],
                         ["Do u deliver to Montgomery?", InquiryDataset.delivery],
                         ["Do u deliver to Durham?", InquiryDataset.delivery],
                         ["Do u deliver out of the NYC?", InquiryDataset.delivery],
                         ["Is there any extra fee for delivering out of the NYC",
                          InquiryDataset.delivery],
                         ["Hi, what's your number, please? I'm interested in Razer Cynosa",
                          InquiryDataset.dataset_search],
                         ["Have you got Razer Cynosa in stock?",
                          InquiryDataset.dataset_search],
                         ["Have you got a black gaming table 150x300?",
                          InquiryDataset.dataset_search],
                         ["Hi, I need a black soft sofa. Have you got any?",
                          InquiryDataset.dataset_search],
                         ["OK, is there ANY black table 150x300?",
                          InquiryDataset.dataset_search],
                         ["Hi, I'm interested in HyperX mouse pad? How much do you want for it?",
                          InquiryDataset.dataset_search],
                         ["Hi, may I please call you up?", InquiryDataset.contact],
                         ["Hey, I need a phone for calls and texting.",
                          InquiryDataset.recommendation],
                         ["Hi, I need a small cellphone for my kid. A cheap one. ",
                          InquiryDataset.recommendation],
                         ["Where can I find you, guys? Have you got any store or something?",
                          InquiryDataset.contact],
                         ["What's the problem with the delivery?",
                          InquiryDataset.delivery],
                         ["Can you ship it to Ukraine?", InquiryDataset.delivery],
                         ["Can you ship it to Russia?", InquiryDataset.delivery],
                         ["Is it possible to ship it to UK?",
                          InquiryDataset.delivery],
                         ["Hi, are you interested in investing?",
                          InquiryDataset.user_interaction_needed],
                         ["What are the ways to pay?",
                          InquiryDataset.checkoutRequest],
                         ["Hi, I wanna buy IPhone 12. John Wick, 2302  Oral Lake Road, Minneapolis.",
                          InquiryDataset.order + InquiryDataset.checkout + InquiryDataset.dataset_search],
                         ["Hi, 5 California and 3 Crunchy Rolls. 1338  Saint Francis Way, Brookfield, pay in cash.",
                          InquiryDataset.dataset_search + InquiryDataset.order + InquiryDataset.checkout],
                         ["What does California roll consist of?",
                          InquiryDataset.dataset_search],
                         ["What is in Califronia roll?",
                          InquiryDataset.dataset_search],
                         ["Should I buy Crunchy Roll if I'm allergic to tuna?",
                          InquiryDataset.dataset_search + InquiryDataset.user_interaction_needed],
                         ["Hey, do you need any shop assistants?",
                          InquiryDataset.user_interaction_needed],
                         ["Hi, the rolls were so awesome! My tongue almost blew up because of the amazing taste."
                          "The sauce was so sweet and tasty!", InquiryDataset.feedback],
                         ["Hi, I need a cheap IPhone. May I pay with cash?",
                          InquiryDataset.checkoutRequest + InquiryDataset.recommendation],
                         ["What's the price of the used MrBeast merch?",
                          InquiryDataset.dataset_search],
                         [
                             "Hi, a quick review of what I received. I got a OnePlus 7 Pro, everything was well packed so it couldn't break down or something"
                             "The phone itself is awesome. I highly recommend the store!", InquiryDataset.feedback],
                         ["Hey, got the rolls. Thanks, they're absolutely delicious.",
                          InquiryDataset.feedback],
                         ["Hi, I've never read either the Lord of the Rings or Harry Potter. What should I get?",
                          InquiryDataset.recommendation],
                         ["Hi, if I order a book today, when will you ship it?",
                          InquiryDataset.delivery],
                         ["Hey, do you have a white office chair?",
                          InquiryDataset.dataset_search],
                         ["Hello, do you need a desinger for your store?",
                          InquiryDataset.user_interaction_needed],
                         ["What is the most interesting book from the new ones?",
                          InquiryDataset.user_interaction_needed],
                         ["How much for the Institute of Stephen King?",
                          InquiryDataset.dataset_search],
                         ["How much do you want for Dale Carnegie How to Win Friends and Influence People?",
                          InquiryDataset.dataset_search],
                         ["Are there any Christmas discounts?",
                          InquiryDataset.welcome],
                         ["Hi, do you have a veteran discount?",
                          InquiryDataset.welcome],
                         ["Is there discount if I buy a couple of clothes?",
                          InquiryDataset.welcome],
                         ["Hi, have you got a store in NYC?",
                          InquiryDataset.contact],
                         ["Should I buy Deathadder V2 or Viper?",
                          InquiryDataset.dataset_search],
                         ["My son's going to school. Do you have any shoes size 9.",
                          InquiryDataset.recommendation],
                         ["My daughter is going to the university. I need nice comfortable shoes.",
                          InquiryDataset.recommendation],
                         ["Hi, how many cores does IPhone's processor have?",
                          InquiryDataset.dataset_search],
                         ["Hey, i need a phone for my parent asap. It has to be cheap and fast.",
                          InquiryDataset.recommendation],
                         ["Hi, what's the phone number?", InquiryDataset.contact],
                         ["I need a red dress for a party. Do you have one?",
                          InquiryDataset.recommendation],
                         ["Do u deliver to Russia? How much time is it gonna take?",
                          InquiryDataset.delivery],
                         ["Okay, then here is my address: 1887  Philadelphia Avenue, Brian C Vasquez. Cash.",
                          InquiryDataset.order + InquiryDataset.checkout],
                         [
                             "Hey, I wanna get the new Gucci dress 2652. My address: 1086  Upton Avenue, 	Bobby M Bukowski. Card.",
                             InquiryDataset.order + InquiryDataset.checkoutRequest + InquiryDataset.dataset_search],
                         ["Hi, do you still have that salvaged Model S 2016?",
                          InquiryDataset.dataset_search],
                         ["Okay, let's deliver it to 2733  Golden Ridge Road, Schenectady, Kevin C Perez",
                          InquiryDataset.checkoutRequest + InquiryDataset.order],
                         ["Hi, what are the methods to checkout?",
                          InquiryDataset.checkoutRequest],
                         ["Hi, I need IPhone 10 for 512 gb Black",
                          InquiryDataset.dataset_search],
                         [
                             "Whassup guys, one pair Converse 70 Low yellow. Address: 	Judith J Knight,Michigan, 884  D Street",
                             InquiryDataset.dataset_search + InquiryDataset.order],
                         ["What's the price of FENTY BEAUTY by Rihanna Match Stix Trio",
                          InquiryDataset.dataset_search],
                         ["Hey, do you have something for my 7-year old daughter?",
                          InquiryDataset.recommendation],
                         ["Hi, what's the size of those Adidas Yeezy limited green?",
                          InquiryDataset.dataset_search],
                         ["Hey, are they soft?",
                          InquiryDataset.user_interaction_needed],
                         ["Do you have 9 size Converse Taylor All Star Mono High-Top?",
                          InquiryDataset.dataset_search],
                         [
                             "Hi, 9 size Converse Taylor All Star Mono High-Top. Address: 	Judith J Knight,Michigan, 884  D Street",
                             InquiryDataset.dataset_search + InquiryDataset.order],
                         ["Hi, they're so soft and sweet! Thank you so much for that!",
                          InquiryDataset.feedback],
                         [
                             "Hi, Idk if they have to be that way, but they're so awesome for their price. It's only 200$ and that's relatively cheap price! So greateful that bought them here.",
                             InquiryDataset.feedback],
                         ["What are the best budget keyboard that you have?",
                          InquiryDataset.recommendation],
                         ["What is the Trust 25652e gamepad made of?",
                          InquiryDataset.dataset_search],
                         ["Okay, my address is: 4460  Centennial Farm Road, MONTAGUE, Massachusetts, David N Cadet",
                          InquiryDataset.order],
                         ["Alright, my address is: 1034  John Calvin Drive, Chicago, IL, 60606, 708-896-8588",
                          InquiryDataset.order],
                         [
                             "Okay, then get it to: 1761  Wildrose Lane, Detroit, Michigan, 48226, 313-805-3148, 	Jason C Clark",
                             InquiryDataset.order],
                         ["What is the size of Wilson Evolution?", InquiryDataset.dataset_search],
                         ["What about the payment methods?", InquiryDataset.checkoutRequest],
                         ["Have you got a cheap leather ball?", InquiryDataset.dataset_search],
                         [
                             "Oh, fuck, that ball is so cool. It bounces really well and I dunk all the time. "
                             "It grips really well and seems it won't wear off in the future. Highly recommend!",
                             InquiryDataset.feedback],
                         ["I'm looking for a ball for my kid. What would you recommend?",
                          InquiryDataset.recommendation]],

                        )

    @staticmethod
    def save(dataset):
        pd.DataFrame(dataset).to_csv(InquiryDataset.dataset_path)
