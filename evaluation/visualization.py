import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from config import Config

def residual_plot(gt, pred):
    """ Plot the residual of the prediction. """
    residuals = gt - pred
    plt.figure(figsize=(8,6))
    plt.title("Residual Plot")
    plt.scatter(gt, residuals, color='black', s=10)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel("True Values")
    plt.ylabel("Residuals")
    plt.savefig(f"{Config.IMAGES_OUTPUT}/residual_plot.png", dpi=300)
    plt.show()
    

def plot_errors(y_true, y_pred):
    """ Plot the distribution of the absolute errors. """
    errors = np.abs(y_true - y_pred)
    sns.histplot(errors, kde=True, bins=30)
    plt.xlabel("Absolute error")
    plt.ylabel("Frequencies")
    plt.title("Distribution of absolute errors")
    plt.savefig(f"{Config.IMAGES_OUTPUT}/residual_plot.png", dpi=300)
    plt.show()
