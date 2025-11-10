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
import trendAnalysis

print("All libraries loaded for "+ __file__)



param = {
    "symbol": "PLTR",
    "start_date": "2020-09-20",
    # "end_date":"2024-03-01",
    "current_estEPS": 0.74,
    # "current_unemploy": 4.3,
    "comment": "None",
    "threshold": 0.05,
    "selected_columns": [
       "label",

        ##################################
        # company Stock fundamentals
        ##################################
        "adjusted close",
        "daily_return",     # this is good
        "volume",
        # "volume_norm",    # regular volume still have better F1
        'Volume_Oscillator',
        "volatility",
        "VWAP",           # Volume Weighted Average Price (10/5/24 VWAP > Volume)
        "high",           # high low is slightly underperforming
        "low",

        ##################################
        # company fundamentals
        ##################################
        'EPS', 
        'estEPS',
        'surprisePercentage',
        'totalRevenue',
        'netIncome',
        
        ##################################
        # Stock Momentum 
        ##################################
        "MACD_Signal",
        "MACD",
        "MACD_Hist",
        "ATR",
        "RSI",
        'Real Upper Band',  # bollinger band (taken out as not that useful)
        'Real Middle Band',
        'Real Lower Band',

        ##################################
        # Monetary market
        ##################################
        "interest",
        "10year",
        "2year",
        "3month",
        'DTWEXBGS',         # DOLLAR INDEX (10/5/24 this not as good as using eur_close)
        'DFEDTARU',         # federal fund rate upper limit
        'BOGMBBM',          # Monetary Base; Reserve Balances
        'eur_close',
        'jpy_close',
        # 'USALOLITONOSTSAM', # Leading Indicators: Composite Leading Indicator: Normalised for United States (in 5/9/24, no data since Jan 24)
      

        ##################################
        # Other Indeces
        ##################################
        "SPY_close",
        "qqq_close",
        "VTWO_close",
        'SPY_stoch',        #the appromiate for SnP Oscillator using slowD (slightly better than SPY_close)
        'calc_spy_oscillator',  # self calculated SnP500 oscillator based on SPY
        'QQQ_stoch',      #the appromiate for NASDAQ Oscillator using slowD (very close but in combo not as good as using QQQ_close)
        'VTWO_stoch',       #the appromiate for Russell Oscillator using slowD (much better)
  
        ##################################
        # General Market fundamentals
        ##################################
        "VIXCLS",           # VIX from FRED
        # "VIXCLS_Y",         # VIX from YFinance
        "DCOILWTICO",       # WTI oil price
        # "unemploy",         # as of 5/9, still no satat for 5/1
        "FEDTARMDLR",       # Longer Run FOMC Summary of Economic Projections for the Fed Funds Rate, Median (probably not useful (in 5/9/24, last data point is 3/20/24)
        # "USEPUINDXD",       # Economic Policy Uncertainty Index for United States
        'UMCSENT',          # consumer sentiment (removed, slightly worse then base (no update since 3/1/24)
        # 'BSCICP03USM665S',  # sentiment OECD   (at 5/9/24, no data since Jan 24)
        'DGORDER',          # DURABLE GOODS ORDER
        'CORESTICKM159SFRBATL',     # Sticky Price Consumer Price Index less Food and Energy (no update in 5 mo, I think this is dead)
        # 'PCE',              # No data since 3/1
        'PCU33443344',      # Producer Price Index by Industry: Semiconductor and Other Electronic Component Manufacturing 
        'SAHMREALTIME',     # sahm unemployment (seems to work better than unemploy)
        'JTSJOL',           # non farm job opening
        'GDP',              # Add GDP
        'FINRA_debit',      # finra debit
    ],
    "window_size": 35,
    "target_size": 15,
    "test_size": 0.2,
    "validation_size": 0.25,
    "batch_size": 32,       # best amongst 25, 32, and 40
    "shuffle": False,       # even though turning it on improves things, should not do it for time series data
    "headcount": 8,         
    "num_layers": 2,            # 2 MUCH BETTER THAN 3
    "dropout_rate": 0.1,
    "learning_rate": 0.0005,
    "num_epochs": 200,
    "down_weight": 1.5,        # tried 1.25 to 1.75. 1,5 still the best
    "scaler_type": 'MinMax',    # MinMax is the best
    "volatility_window": 13,     # 13 is best among [4, 7, 10, 13, 16]
    "bband_time_period": 20,    # Bollinger band time period (changing it doesn't seem to make a dent)
    "training_set_size": 570,
    "validation_set_size": 195,
    "l1_lambda": 1e-6,          # This better than 0.5e-6
    "l2_weight_decay": 1.6e-5,  # 1.6 seems best
    "embedded_dim": 64         # default to 64
}

#######
# MAIN LOOP enters here

logging.basicConfig(level=logging.INFO)



# elements = [0.007, 0.01, 0.013]
# for item in elements:
#     param["threshold"] = 0.01

trendAnalysis.load_data_to_cache(trendConfig.config, param)
 
param["comment"]='baseline for PLTR'
trendAnalysis.analyze_trend(trendConfig.config, param, True)