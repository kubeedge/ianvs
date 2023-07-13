import logging
import numpy as np
import joblib
import matplotlib.pyplot as plt

# plt.style.use(['science', 'ieee'])
logging.basicConfig(level=logging.INFO)
from tree import TForest_Tree,AOA_dataset,Tree
import os
import statsmodels.api as sm
import scienceplots
from tqdm import tqdm
from torch.utils.data import DataLoader
# plt.style.use(['science', 'ieee'])

import torch
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
device  = torch.device('cuda')
"""

"""
def test_main():
    tree_list = joblib.load('initial_tree.joblib')

if __name__ == '__main__':
