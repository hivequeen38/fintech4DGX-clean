## inference only main

import sys
import pandas as pd
from pandas import DataFrame
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score

from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from itertools import cycle
import seaborn as sns
import torch.nn as nn
import json
from datetime import datetime
import logging
import os
import fetchBulkData
import etl
import processPrediction
import trendConfig
import trendAnalysis

print("All libraries loaded for "+ __file__)

# first download the data as normal
### IMPORTANT these can only change to match the trained Model

def did_value_change( df: DataFrame, column_name: str) -> bool:
    # Check if the value in 'Column1' in the last row is different from the previous row
    value_changed = False
    if len(df) > 1:  # Ensure there are at least two rows to compare
        value_changed = df[column_name].iloc[-1] != df[column_name].iloc[-2]
        # print("Has the value changed in the last row compared to the previous row?", value_changed)
    else:
        print("Not enough data to compare two rows.")
    return value_changed

def doInference():
    random.seed(42)  # Python's built-in random lib
    np.random.seed(42)  # Numpy lib
    torch.manual_seed(42)  # PyTorch

    # If you are using CUDA
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)  # if you are using multi-GPU.

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config = trendConfig.config
    df, num_data_points, display_date_range = trendAnalysis.download_data(config, param)

    raw_df = df.copy()      # preserve the original, there the end of it contains the data needed for prediction

    data_changed = did_value_change(raw_df, 'close')
    if (data_changed):
        # then setup labels
        trendAnalysis.calculate_label(df, param)
        selected_columns = param['selected_columns']

        df = df[selected_columns]   # only keep those slected features defined in the param

        model = torch.load('model.pth')     #!!! Need to change it so get best performing model
        currentDateTime = datetime.now()
        trendAnalysis.make_prediciton(model, df, param, selected_columns, currentDateTime, 'NVDA')
    else:
        print('Today data not yet ready. close has not changed from previous day')

