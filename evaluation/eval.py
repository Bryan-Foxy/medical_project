"""
Author:     FOZAME ENDEZOUMOU Armand Bryan 

This code is for the evaluation for our soliution, 
we will compare the original VHS to the predicted one.
Also compare the diameters.

We will perform MAE and MSE.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the data
class Evaluator:
    """ This class will evaluate the VHS predictions model.
    It will compare the predicted VHS score to the ground truth VHS score.
    It will also compare the predicted major and minor diameters to the ground truth major and minor diameters.
    """
    def __init__(self, model_heart, model_vertebrae):
        self.model_heart = model_heart
        self.model_vertebrae = model_vertebrae
    
    def _load_csv_data(self, path):
        """ Load the data from a CSV file"""
        data = pd.read_csv(path)
        return data
    
    def _load_data(self, path):
        """ Load the data from a file or folder """
        if os.path.isdir(path):
            data = []
            for file in tqdm(os.listdir(path)):
                data.append(os.path.join(path, file))
            return data
        elif os.path.isfile(path):
            return [path]
        else:
            raise ValueError("Invalid path")
    

