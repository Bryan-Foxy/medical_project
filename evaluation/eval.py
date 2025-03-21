"""
Author:     FOZAME ENDEZOUMOU Armand Bryan 

This code is for the evaluation for our soliution, 
we will compare the original VHS to the predicted one.
Also compare the diameters.

We will perform MAE and MSE.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.vhs import VHS

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
        data = pd.read_excel(path)
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
    
    def _match_csv2data(self, csv_data, data, inverse = False):
        """ 
        This function will match the csv data to the data.
        It will return the matched data.
        - csv_data: pd.DataFrame
        - data: list
        - inverse: bool
        """
        if not inverse:
            for _, row in csv_data.iterrows():
                id_value = str(row["\nFilename"])
                if id_value in data:
                    return id_value
            return None
        else:
            matched_row = csv_data[csv_data["\nFilename"] == data]
            return matched_row
    
    def evaluate(self, csv_path, data_path):
        """ Perform the evaluation"""
        # Initialize dictionaries to store the values
        pred_dict = {"id": [],
                     "vhs_score": [],
                     "major_diameter": [],
                     "minor_diameter": []}
        gt_dict = {"id": [],
                   "vhs_score": [],
                   "major_diameter": [],
                   "minor_diameter": []}
        # Prepare the data
        csv_dset = self._load_csv_data(csv_path)
        print(csv_dset.columns)
        dset_images = self._load_data(data_path)
        images_path = [path for path in dset_images if any(val in path for val in csv_dset["\nFilename"])]

        # Initialize and perform the VHS algorithm
        for image_path in tqdm(images_path, desc = "Performing VHS algorithm"):
            id = self._match_csv2data(csv_dset, image_path)
            gt_dict["id"].append(id)
            pred_dict["id"].append(id)
            gt_dict["vhs_score"].append(csv_dset[csv_dset["\nFilename"] == id]["\nVHS/IB"].values[0])
            vhs = VHS(image_path = image_path, model_heart = self.model_heart, model_vertebrae = self.model_vertebrae)
            M, m, pred_vhs_score, output_path = vhs.perform_vhs()
            pred_dict["vhs_score"].append(pred_vhs_score)
        
        # Convert vhs list to numpy array
        gt_dict["vhs_score"] = np.array(gt_dict["vhs_score"])
        pred_dict["vhs_score"] = np.array(pred_dict["vhs_score"])

        # Compute the MAE and MSE
        print("VHS algorithm performed successfully")
        print(f"The model was evaluated on {len(gt_dict['id'])} images")
        mse = np.mean((gt_dict["vhs_score"] - pred_dict["vhs_score"]) **2)
        mae = np.mean(np.abs(gt_dict["vhs_score"] - pred_dict["vhs_score"]))
        print("VHS evaluation results:")
        print("="*40)
        print(f"-> Mean Squared Error:{mse}")
        print(f"-> Mean Absolute Error:{mae}")

if __name__ == "__main__":
    model_heart = "/Users/armandbryan/Documents/aivancity/PGE5/Medical AI/project/models/heart_segmentation_model.h5"
    model_vertebrae = "/Users/armandbryan/Documents/aivancity/PGE5/Medical AI/project/models/vertebrae_yolo.pt"
    csv_path = "/Users/armandbryan/Documents/aivancity/PGE5/Medical AI/project/datas/VERTEBRAE_HEART_SCALES_ANNOTATIONS_MT.xlsx"
    data_path = "/Users/armandbryan/Documents/aivancity/PGE5/Medical AI/project/datas/valid"
    evaluator = Evaluator(model_heart = model_heart, model_vertebrae = model_vertebrae)
    evaluator.evaluate(csv_path = csv_path, data_path = data_path)






    

