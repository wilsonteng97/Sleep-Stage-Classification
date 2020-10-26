import numpy as np
import pandas as pd

from config import Config


class DataExtractor():
    """ 
    DataExtractor class contains functions that extract features from x and returns a DataFrame.
    """

    def __init__(self, x, y, config=Config()):
        assert x.shape[1] == 3000
        assert x.shape[0] == y.shape[0]
        
        self.x = x
        self.y = y

    """ TODO : Function to generate pandas DataFrame with all the features stipulated in config. """
    def generateDF(self):
        print("TODO")

    """ TODO : Dummy function example. """
    def calculateFeature1(self):
        return np.mean(self.x)
