"""
This module adds the contractions for inquiry processing for InquiryConverter module.
"""
from contractions import ContractionsJSON
if __name__ == '__main__':
    print("The form is: \n [contraction];replacement")
    contractions = ContractionsJSON()
    contractions.initDictionary()
    print(contractions.path)
    while True:
        try:
            a = input()
            b = a.split(";")
            contraction, replacement = b[0], b[1]
            contractions.add(contraction, replacement)
            contractions.saveJSON()

        except Exception as e:
            print(e)
