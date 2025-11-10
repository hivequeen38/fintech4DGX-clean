# import sys
# import pandas as pd
# from pandas import DataFrame
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import TensorDataset, DataLoader

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
# from sklearn.metrics import roc_curve, auc, roc_auc_score
# from sklearn.preprocessing import label_binarize
# import matplotlib.pyplot as plt
# from itertools import cycle
# import seaborn as sns
# import torch.nn as nn
# import json
# from datetime import datetime
# import os
# import fetchBulkData
# import etl
import logging      
import trendConfig
import trendAnalysisBlackBox

print("All libraries loaded for "+ __file__)



param = {
    "symbol": "NVDA",
    "start_date": "2020-06-17",
    "end_date":"2024-03-01",
    "comment": "None",
    "threshold": 0.05,
    "selected_columns": [
       "label",
        "adjusted close",
        "daily_return",     # this is good
        # "volume",
        "volume_norm",    # regular volume still have better F1
        "volatility",
        # "VWAP",           # not useful compare to just vol and adjusted close
        # "high",           # high low is slightly underperforming
        # "low",
        # "interest",
        "SPY_close",
        "10year",
        "2year",
        # "3month",
        "VIXCLS",
        "MACD_Signal",
        "MACD",
        "MACD_Hist",
        "ATR",
        "RSI",
        "DCOILWTICO",
        "unemploy",       # as of 5/9, still no satat for 5/1
        "FEDTARMDLR",       ##### probably not useful (in 5/9/24, last data point is 3/20/24)
        "USEPUINDXD",       # Economic Policy Uncertainty Index for United States
        # 'DTWEXBGS',         # DOLLAR INDEX removed, still seems worse even accounted for not reading stale files
        'UMCSENT',          # removed, slightly worse then base (no update since 3/1/24)
        # 'BSCICP03USM665S', # sentiment OECD   (at 5/9/24, no data since Jan 24)
        'DGORDER',          # ADD DURABLE GOODS ORDER
        # 'USALOLITONOSTSAM',    # in 5/9/24, no data since Jan 24
        'DFEDTARU',
        'CORESTICKM159SFRBATL',
        'PCE',              # No data since 3/1
        'Real Upper Band',
        'Real Middle Band',
        'Real Lower Band',
        'EPS', 
        # 'estEPS',
        'surprisePercentage',
        'totalRevenue',
        'netIncome',
        'PCU33443344',
    ],
    "window_size": 35,
    "target_size": 15,
    "test_size": 0.2,
    "validation_size": 0.25,
    "batch_size": 32,       # best amongst 25, 32, and 40
    "shuffle": False,
    "headcount": 5,         # 2 is better than s 1 (3 is best if divisible)
    "num_layers": 2,
    "dropout_rate": 0.1,
    "learning_rate": 0.001,
    "num_epochs": 200,
    "down_weight": 1.50,        # tried 1.25 to 1.75. 1,5 still the best
    "scaler_type": 'MinMax',    # MinMax is the best
    "volatility_window": 13,     # 13 is best among [4, 7, 10, 13, 16]
    "bband_time_period": 20,    # Bollinger band time period (changing it doesn't seem to make a dent)
    "training_set_size": 570,
    "validation_set_size": 195,
}

#######
# MAIN LOOP enters here

logging.basicConfig(level=logging.INFO)



elements = ["2024-03-06", "2024-03-07", "2024-03-08", 
            "2024-03-11", "2024-03-12", "2024-03-13", "2024-03-14", "2024-03-15",
            "2024-03-18", "2024-03-19", "2024-03-20", "2024-03-21", "2024-03-22",
            "2024-03-25", "2024-03-26", "2024-03-27", "2024-03-28", "2024-03-29",
            "2024-03-30", "2024-03-31", "2024-04-01", "2024-04-02", "2024-04-03",
            "2024-04-06", "2024-04-07", "2024-04-08", "2024-04-09", "2024-04-10",
            "2024-04-13", "2024-04-14", "2024-04-15", "2024-04-16", "2024-04-17",
            "2024-04-20", "2024-04-21", "2024-04-22", "2024-04-23", "2024-04-24",
            "2024-04-27", "2024-04-28", "2024-04-29", "2024-04-30", "2024-05-01",
            "2024-05-04", "2024-05-05", "2024-05-06", "2024-05-07", "2024-05-08",
            "2024-05-11", "2024-05-12", "2024-05-13", "2024-05-14", "2024-05-15",
            "2024-05-18", "2024-05-19", "2024-05-20", "2024-05-21", "2024-05-22",
            "2024-05-25", "2024-05-26", "2024-05-27", "2024-05-28", "2024-05-29",
            "2024-06-03", "2024-06-04", "2024-06-05", "2024-06-06", "2024-06-07",
            "2024-06-10", "2024-06-11", "2024-06-12", "2024-06-13", "2024-06-14",
            "2024-06-17", "2024-06-18", "2024-06-19", "2024-06-20"
            ]
for item in elements:
    param["end_date"] = item
    param["comment"]='stepwise testing now for end_date= ' + item
    trendAnalysisBlackBox.analyze_trend(trendConfig.config, param)