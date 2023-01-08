import numpy as np
class PandasService:
    @staticmethod
    def pandasAr_tonumpyAr(a) -> np.ndarray:
        """
        Pandas numpy array is a string. To convert to numpy array we use this method.
        Converts '[1 0 1 0 0 0 0 0 0]' to 'array([[1, 0, 1, 0, 0, 0, 0, 0, 0]]'
        :param a: csv string or list of csv strings
        :return: np.ndarray
        """
        assert(type(a) is str or type(a) is list)
        if type(a) is list:
            for i in list:
                assert(i is str)
            # let's select the first value and see how many values does it have 
            b = a[0].split('[')[1].split(']')[0].split(' ')
             # initialize the numpy array 
            c = np.zeros([len(len(b))], 
                         dtype=np.int16)
            for j in list:
                b = a.split('[')[1].split(']')[0].split(' ')
                for i, el in enumerate(b):
                    c[i] = int(el)
            return c
        else:
            
            b = a.split('[')[1].split(']')[0].split(' ')
            c = np.zeros([len(b)],dtype=np.int16)
            for i, el in enumerate(b):
                c[i] = int(el)
            return c
    