import numpy as np
import pandas as pd
from pandas import DataFrame
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from sklearn.preprocessing import StandardScaler

from alpha_vantage.timeseries import TimeSeries
import alpha_vantage.fundamentaldata as av_fund
import json
import numpy as np
# from datetime import datetime
import fetchBulkData
import etl

print("All libraries loaded")

config = {
    "alpha_vantage": {
        "key": "H896AQT2GYE4ZO8Z", # Claim your free API key here: https://www.alphavantage.co/support/#api-key
        "symbol": "BTC",
        "outputsize": "full",
        "key_adjusted_close": "5. adjusted close",
        "key_volume": "6. volume",
        "key_high": "2. high",
        "key_low": "3. low",
        "SPY_symbol": "SPY",
        "url": "https://www.alphavantage.co/query"
    },
    "data": {
        "window_size": 20,      # I thhnk this is the sequence size
        "train_split_size": 0.75,
    },
    "plots": {
        "xticks_interval": 90, # show a date every 90 days
        "color_actual": "#001f3f",
        "color_train": "#3D9970",
        "color_val": "#0074D9",
        "color_pred_train": "#3D9970",
        "color_pred_val": "#0074D9",
        "color_pred_test": "#FF4136",
    },
    "model": {
        "num_lstm_layers": 1,   # change to single layer
        "lstm_size": 36,
        "dropout": 0.2,
    },
    "training": {
        "device": "cpu", # "cuda" or "cpu"
        "batch_size": 64,
        "num_epoch": 100,
        "learning_rate": 0.01,
        "scheduler_step_size": 40,
    },
    "eodhd": {

    }
}

def download_data(config):
    # ts = TimeSeries(key=config["alpha_vantage"]["key"])
    # data, meta_data = ts.get_daily_adjusted(config["alpha_vantage"]["symbol"], outputsize=config["alpha_vantage"]["outputsize"])
    # spy_data, spy_meta_data = ts.get_daily_adjusted(config["alpha_vantage"]["SPY_symbol"], outputsize=config["alpha_vantage"]["outputsize"])
    

    # data_date = [date for date in data.keys()]
    # data_date.reverse()

    # data_close_price = [float(data[date][config["alpha_vantage"]["key_adjusted_close"]]) for date in data.keys()]
    # data_close_price.reverse()
    # data_close_price = np.array(data_close_price)

    # # duplicate the above logic to get the volume
    # data_volume = [float(data[date][config["alpha_vantage"]["key_volume"]]) for date in data.keys()]
    # data_volume.reverse()
    # data_volume = np.array(data_volume)

    # # now get the high
    # data_high = [float(data[date][config["alpha_vantage"]["key_high"]]) for date in data.keys()]
    # data_high.reverse()
    # data_high = np.array(data_high)

    # # now get the low
    # data_low = [float(data[date][config["alpha_vantage"]["key_low"]]) for date in data.keys()]
    # data_low.reverse()
    # data_low = np.array(data_low)

    # # handle SPY to represent market condition
    # spy_close_price = [float(spy_data[date][config["alpha_vantage"]["key_adjusted_close"]]) for date in data.keys()]
    # spy_close_price.reverse()
    # spy_close_price = np.array(spy_close_price)

    # Use the bulk fetch code to get all the needed stuff in one call
    df = fetchBulkData.fetch_all_data(config)
    # print(df.head)


    ###########################
    # Now calculate a colu called gain
    # df['gain'] = (df['adjusted close'] / df['adjusted close'].shift(1))

    df = etl.fill_data(df)
    print('>DF is cleaned up after ETL')
    print(df.head)
    # work on fundamental data
    # data, meta_data = av_fund.FundamentalData

    # filter it so data is only 2021-06-17 (valid for BTC)
    start_date = '2021-06-17'
    df = df[df['date'] >= start_date]

    num_data_points = len(df)
    display_date_range = "from " + df['date'].iloc[0] + " to " + df['date'].iloc[num_data_points-1]
    print("Number data points", num_data_points, display_date_range)

    # merge all the features together
    print('shape of feature data in a df: ', df.shape)
    return df, num_data_points, display_date_range

df, num_data_points, display_date_range = download_data(config)

# now we have to construct data_date (list), feature data from the df
data_date = df['date'].tolist()

#######################################################
# add more columns if we want to have more features
# also needs to adjust the normailized version
#
selected_columns = [
    # 'gain',
    #'adjusted close',
    # 'volume',
    # 'high', 
    # 'low',
    # 'interest', 
    # '10year', 
    'VIXCLS', 
    # 'unemploy',
    'MACD_Signal',
    'MACD',
    'MACD_Hist',
    # 'ATR',
    'RSI',
    # 'DTWEXBGS',
    # 'UMCSENT',
    # 'DCOILWTICO',
    # 'BSCICP03USM665S',
    'BTC_close'
    ]

num_of_features: int = len(selected_columns)
features_data = df[selected_columns].to_numpy()

# plot price
# fig = figure(figsize=(25, 5), dpi=80)
# fig.patch.set_facecolor((1.0, 1.0, 1.0))
# plt.plot(data_date, features_data[:,0], color=config["plots"]["color_actual"])
# xticks = [data_date[i] if ((i%config["plots"]["xticks_interval"]==0 and (num_data_points-i) > config["plots"]["xticks_interval"]) or i==num_data_points-1) else None for i in range(num_data_points)] # make x ticks nice
# x = np.arange(0,len(xticks))
# plt.xticks(x, xticks, rotation='vertical')
# plt.title("Daily Closing Price for " + config["alpha_vantage"]["symbol"] + ", " + display_date_range)
# # plt.grid(b= False, which='major', axis='y', linestyle='--') # this line as a bug about 'b' can probably do away with it
# plt.grid(which='major', axis='y', linestyle='--') # this line as a bug about 'b' can probably do away with it
# plt.show()

# now plot volume
# fig = figure(figsize=(25, 5), dpi=80)
# fig.patch.set_facecolor((1.0, 1.0, 1.0))
# plt.plot(data_date, features_data[:,1], color=config["plots"]["color_actual"])
# xticks = [data_date[i] if ((i%config["plots"]["xticks_interval"]==0 and (num_data_points-i) > config["plots"]["xticks_interval"]) or i==num_data_points-1) else None for i in range(num_data_points)] # make x ticks nice
# x = np.arange(0,len(xticks))
# plt.xticks(x, xticks, rotation='vertical')
# plt.title("Daily Volume for " + config["alpha_vantage"]["symbol"] + ", " + display_date_range)
# # plt.grid(b= False, which='major', axis='y', linestyle='--') # this line as a bug about 'b' can probably do away with it
# plt.grid(which='major', axis='y', linestyle='--') # this line as a bug about 'b' can probably do away with it
# plt.show()

class OLDNormalizer():
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit_transform(self, x):
        self.mu = np.mean(x, axis=(0), keepdims=True)
        self.sd = np.std(x, axis=(0), keepdims=True)
        normalized_x = (x - self.mu)/self.sd
        return normalized_x

    def inverse_transform(self, x):
        return (x*self.sd) + self.mu

class Normalizer():
    def __init__(self):
        self.min = None
        self.max = None

    def fit_transform(self, x):
        self.min = np.min(x)
        self.max = np.max(x)
        return (x - self.min) / (self.max - self.min)

    def inverse_transform(self, x):
        return x * (self.max - self.min) + self.min

class MaxAbsScaling():
    def __init__(self):
        self.max_abs_value = None

    def fit_transform(self, x):
        self.max_abs_value = np.max(np.abs(x))
        normalized_x = x / self.max_abs_value
        return normalized_x

    def inverse_transform(self, x):
        return x * self.max_abs_value
    
# normalize
# note the data are restacked here so must do this every time a new feature is added
#
# high_scaler = StandardScaler()
# df['high'] = high_scaler.fit_transform(df[['high']])
# low_scaler = StandardScaler()
# df['low'] = low_scaler.fit_transform(df[['low']])
# close_scaler = StandardScaler()
# df['adjusted close'] = close_scaler.fit_transform(df[['adjusted close']])

# volume_scaler = StandardScaler()
# df['volume'] = volume_scaler.fit_transform(df[['volume']])
# spy_scaler = StandardScaler()
# df['SPY_close'] = spy_scaler.fit_transform(df[['SPY_close']])
# interest_scaler = StandardScaler()
# df['interest'] = interest_scaler.fit_transform(df[['interest']])
# tenyear_scaler = StandardScaler()
# df['10year'] = tenyear_scaler.fit_transform(df[['10year']])
# vxx_scaler = StandardScaler()
# df['vxx_close'] = vxx_scaler.fit_transform(df[['vxx_close']])
# unemploy_scaler = StandardScaler()
# df['unemploy'] = unemploy_scaler.fit_transform(df[['unemploy']])

# normalized_feature_data = df[selected_columns].to_numpy()

# high_scaler = Normalizer()
# n_data_high_price = high_scaler.fit_transform(features_data[:,2])
# low_scaler = Normalizer()
# n_data_low_price = low_scaler.fit_transform(features_data[:,3])
# gain_scaler = Normalizer()
# n_data_gain = gain_scaler.fit_transform(features_data[:,0])
# price_scaler = Normalizer()
# n_data_close_price = price_scaler.fit_transform(features_data[:,0])
# volume_scaler = Normalizer()
# n_data_volume = volume_scaler.fit_transform(features_data[:,1])
# spy_scaler = Normalizer()
# n_data_spy_price = spy_scaler.fit_transform(features_data[:,4])
# interest_scaler = Normalizer()
# n_data_interest = interest_scaler.fit_transform(features_data[:,1])
# tenyear_scaler = Normalizer()
# n_data_10year = tenyear_scaler.fit_transform(features_data[:,2])
vix_scaler = MaxAbsScaling()
n_data_vix = vix_scaler.fit_transform(features_data[:,0])
# unemploy_scaler = Normalizer()
# n_data_unemploy = unemploy_scaler.fit_transform(features_data[:,4])
macd_Signal_scaler = MaxAbsScaling()
n_data_macd_Signal = macd_Signal_scaler.fit_transform(features_data[:,1])
macd_scaler = MaxAbsScaling()
n_data_macd = macd_scaler.fit_transform(features_data[:,2])
macd_Histo_scaler = MaxAbsScaling()
n_data_macd_Hist = macd_Histo_scaler.fit_transform(features_data[:,3])
# atr_scaler = Normalizer()
# n_data_atr = atr_scaler.fit_transform(features_data[:,6])
rsi_scaler = MaxAbsScaling()
n_data_rsi = rsi_scaler.fit_transform(features_data[:,4])
# DTWEXBGS_scaler = Normalizer()
# n_data_usd = DTWEXBGS_scaler.fit_transform(features_data[:,8])
# n_UMCSENT_scaler = Normalizer()
# n_data_UMCSENT = n_UMCSENT_scaler.fit_transform(features_data[:,9])
# n_wti_scaler = Normalizer()
# n_data_wti = n_wti_scaler.fit_transform(features_data[:,10])
# n_bts_scaler = Normalizer()
# n_data_bts = n_bts_scaler.fit_transform(features_data[:,11])
n_btc_scaler = MaxAbsScaling()
n_data_btc = n_btc_scaler.fit_transform(features_data[:,5])

# IMPORTANT make sure number of feature matches that in the config
normalized_feature_data = np.column_stack((
    n_data_btc,
    # n_data_gain,
    # n_data_close_price, 
    # n_data_volume, 
    # n_data_high_price, 
    # n_data_low_price, 
    # n_data_interest, 
    # n_data_10year, 
    n_data_vix, 
    # n_data_unemploy,
    n_data_macd_Signal,
    n_data_macd,
    n_data_macd_Hist,
    # n_data_atr,
    n_data_rsi
    # n_data_usd,
    # n_data_UMCSENT,
    # n_data_wti,
    ))

# def prepare_data_x(x, window_size):
#     # perform windowing
#     n_row = x.shape[0] - window_size + 1
#     output = np.lib.stride_tricks.as_strided(x, shape=(n_row, window_size), strides=(x.strides[0], x.strides[0]))
#     return output[:-1], output[-1]

def prepare_data_x(x, window_size):
    # x is now a 2D array (num_samples, num_features)
    num_samples, num_features = x.shape
    n_row = num_samples - window_size + 1

    # Create 3D array of shape (n_row, window_size, num_features)
    output = np.lib.stride_tricks.as_strided(
        x, 
        shape=(n_row, window_size, num_features), 
        strides=(x.strides[0], x.strides[0], x.strides[1])
    )
    return output[:-1], output[-1]



def prepare_data_y(x, window_size):
    # # perform simple moving average
    # output = np.convolve(x, np.ones(window_size), 'valid') / window_size

    # use the next day as label
    output = x[window_size:]
    return output



data_x, data_x_unseen = prepare_data_x(normalized_feature_data, window_size=config["data"]["window_size"])
data_y = prepare_data_y(normalized_feature_data[:,0], window_size=config["data"]["window_size"])

# Assuming 'data' is your multi-feature dataset as a 2D NumPy array
# Each row in 'data' is a time step, and each column is a different feature

# window_size = 5  # Example window size
# X, last_sequence = prepare_data_x(data, window_size)
# y = prepare_data_y(data[:, target_feature_index], window_size)  # target_feature_index is the index of the feature you are predicting


# split dataset

split_index = int(data_y.shape[0]*config["data"]["train_split_size"])
data_x_train = data_x[:split_index]
data_x_val = data_x[split_index:]
data_y_train = data_y[:split_index]
data_y_val = data_y[split_index:]

# prepare data for plotting

to_plot_data_y_train = np.zeros(num_data_points)
to_plot_data_y_val = np.zeros(num_data_points)

to_plot_data_y_train[config["data"]["window_size"]:split_index+config["data"]["window_size"]] = n_btc_scaler.inverse_transform(data_y_train)
to_plot_data_y_val[split_index+config["data"]["window_size"]:] = n_btc_scaler.inverse_transform(data_y_val)

to_plot_data_y_train = np.where(to_plot_data_y_train == 0, None, to_plot_data_y_train)
to_plot_data_y_val = np.where(to_plot_data_y_val == 0, None, to_plot_data_y_val)

## plots

fig = figure(figsize=(25, 5), dpi=80)
fig.patch.set_facecolor((1.0, 1.0, 1.0))
plt.plot(data_date, to_plot_data_y_train, label="Prices (train)", color=config["plots"]["color_train"])
plt.plot(data_date, to_plot_data_y_val, label="Prices (validation)", color=config["plots"]["color_val"])
xticks = [data_date[i] if ((i%config["plots"]["xticks_interval"]==0 and (num_data_points-i) > config["plots"]["xticks_interval"]) or i==num_data_points-1) else None for i in range(num_data_points)] # make x ticks nice
x = np.arange(0,len(xticks))
plt.xticks(x, xticks, rotation='vertical')
plt.title("Daily close prices for " + config["alpha_vantage"]["symbol"] + " - showing training and validation data")
# plt.grid(b=None, which='major', axis='y', linestyle='--')
plt.grid(which='major', axis='y', linestyle='--')
plt.legend()
plt.show()

# configure data ready for training
#
class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        # comment out next line as already in the right shape
        # x = np.expand_dims(x, config['model']['input_size']-1) 
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])

dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
dataset_val = TimeSeriesDataset(data_x_val, data_y_val)

print("Train data shape", dataset_train.x.shape, dataset_train.y.shape)
print("Validation data shape", dataset_val.x.shape, dataset_val.y.shape)

train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=False)
val_dataloader = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=False)

# define the LSTM model
#
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size=32, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.linear_1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_layer_size, hidden_size=self.hidden_layer_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        # self.linear_2 = nn.Linear(num_layers*hidden_layer_size, output_size)
        self.linear_2 = nn.Linear(hidden_layer_size, output_size)
        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)

    def OLDforward(self, x):
        batchsize = x.shape[0]

        # layer 1
        x = self.linear_1(x)
        x = self.relu(x)
        
        # LSTM layer
        lstm_out, (h_n, c_n) = self.lstm(x)

        # reshape output from hidden cell into [batch, features] for `linear_2`
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1) 
        
        # layer 2
        x = self.dropout(x)
        predictions = self.linear_2(x)
        return predictions[:,-1]

    def forward(self, x):
        batch_size, seq_len, num_features = x.shape

        # Reshape x to 2D tensor for the linear layer
        x = x.view(batch_size * seq_len, num_features)

        # Apply the linear layer
        x = self.linear_1(x)

        # Apply activation function (ReLU)
        x = self.relu(x)

        # Reshape x back to 3D tensor for LSTM
        x = x.view(batch_size, seq_len, -1)

        # LSTM layer
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Reshape output from hidden cell into [batch, features] for `linear_2`
        # Assuming you want to use the hidden state from the last LSTM layer
        x = h_n[-1]

        # Apply dropout
        x = self.dropout(x)

        # Final linear layer
        predictions = self.linear_2(x)
        
        # Remove the extra dimension
        predictions = predictions.squeeze(-1)

        return predictions

    
    
def run_epoch(dataloader, is_training=False):
    epoch_loss = 0

    if is_training:
        model.train()
    else:
        model.eval()

    for idx, (x, y) in enumerate(dataloader):
        if is_training:
            optimizer.zero_grad()

        batchsize = x.shape[0]

        x = x.to(config["training"]["device"])
        y = y.to(config["training"]["device"])

        out = model(x)
        loss = criterion(out.contiguous(), y.contiguous())

        if is_training:
            loss.backward()
            optimizer.step()

        epoch_loss += (loss.detach().item() / batchsize)

    lr = scheduler.get_last_lr()[0]

    return epoch_loss, lr

train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=True)
val_dataloader = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=True)

model = LSTMModel(input_size=num_of_features, hidden_layer_size=config["model"]["lstm_size"], num_layers=config["model"]["num_lstm_layers"], output_size=1, dropout=config["model"]["dropout"])
model = model.to(config["training"]["device"])

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"], betas=(0.9, 0.98), eps=1e-9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["training"]["scheduler_step_size"], gamma=0.1)

for epoch in range(config["training"]["num_epoch"]):
    loss_train, lr_train = run_epoch(train_dataloader, is_training=True)
    loss_val, lr_val = run_epoch(val_dataloader)
    scheduler.step()
    
    print('Epoch[{}/{}] | loss train:{:.6f}, test:{:.6f} | lr:{:.6f}'
              .format(epoch+1, config["training"]["num_epoch"], loss_train, loss_val, lr_train))
    
# here we re-initialize dataloader so the data doesn't shuffled, so we can plot the values by date

train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=False)
val_dataloader = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=False)

model.eval()

# predict on the training data, to see how well the model managed to learn and memorize

predicted_train = np.array([])

for idx, (x, y) in enumerate(train_dataloader):
    x = x.to(config["training"]["device"])
    out = model(x)
    out = out.cpu().detach().numpy()
    predicted_train = np.concatenate((predicted_train, out))

# predict on the validation data, to see how the model does

predicted_val = np.array([])

for idx, (x, y) in enumerate(val_dataloader):
    x = x.to(config["training"]["device"])
    out = model(x)
    out = out.cpu().detach().numpy()
    predicted_val = np.concatenate((predicted_val, out))

# prepare data for plotting

""" to_plot_data_y_train_pred = np.zeros(num_data_points)
to_plot_data_y_val_pred = np.zeros(num_data_points)

to_plot_data_y_train_pred[config["data"]["window_size"]:split_index+config["data"]["window_size"]] = scaler.inverse_transform(predicted_train)
to_plot_data_y_val_pred[split_index+config["data"]["window_size"]:] = scaler.inverse_transform(predicted_val)

to_plot_data_y_train_pred = np.where(to_plot_data_y_train_pred == 0, None, to_plot_data_y_train_pred)
to_plot_data_y_val_pred = np.where(to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred)

# plots

fig = figure(figsize=(25, 5), dpi=80)
fig.patch.set_facecolor((1.0, 1.0, 1.0))
plt.plot(data_date, data_close_price, label="Actual prices", color=config["plots"]["color_actual"])
plt.plot(data_date, to_plot_data_y_train_pred, label="Predicted prices (train)", color=config["plots"]["color_pred_train"])
plt.plot(data_date, to_plot_data_y_val_pred, label="Predicted prices (validation)", color=config["plots"]["color_pred_val"])
plt.title("Compare predicted prices to actual prices")
xticks = [data_date[i] if ((i%config["plots"]["xticks_interval"]==0 and (num_data_points-i) > config["plots"]["xticks_interval"]) or i==num_data_points-1) else None for i in range(num_data_points)] # make x ticks nice
x = np.arange(0,len(xticks))
plt.xticks(x, xticks, rotation='vertical')
plt.grid(which='major', axis='y', linestyle='--')
plt.legend()
plt.show() """

# prepare data for plotting the zoomed in view of the predicted prices (on validation set) vs. actual prices

to_plot_data_y_val_subset = n_btc_scaler.inverse_transform(data_y_val)
to_plot_predicted_val = n_btc_scaler.inverse_transform(predicted_val)
to_plot_data_date = data_date[split_index+config["data"]["window_size"]:]

# plots

fig = figure(figsize=(25, 5), dpi=80)
fig.patch.set_facecolor((1.0, 1.0, 1.0))
plt.plot(to_plot_data_date, to_plot_data_y_val_subset, label="Actual prices", color=config["plots"]["color_actual"])
plt.plot(to_plot_data_date, to_plot_predicted_val, label="Predicted prices (validation)", color=config["plots"]["color_pred_val"])
plt.title("Zoom in to examine predicted price on validation data portion")
xticks = [to_plot_data_date[i] if ((i%int(config["plots"]["xticks_interval"]/5)==0 and (len(to_plot_data_date)-i) > config["plots"]["xticks_interval"]/6) or i==len(to_plot_data_date)-1) else None for i in range(len(to_plot_data_date))] # make x ticks nice
xs = np.arange(0,len(xticks))
plt.xticks(xs, xticks, rotation='vertical')
# plt.grid(b=None, which='major', axis='y', linestyle='--')
plt.grid(which='major', axis='y', linestyle='--')
plt.legend()
plt.show()

# predict the closing price of the next trading day

model.eval()

#!!! DEBUG
#
print('shape of data unseen before the prediction: ', data_x_unseen.shape)

# x = torch.tensor(data_x_unseen).float().to(config["training"]["device"]).unsqueeze(0).unsqueeze(2) # this is the data type and shape required, [batch, sequence, feature]
x = torch.tensor(data_x_unseen).float().to(config["training"]["device"]).unsqueeze(0) # this is the data type and shape required, [batch, sequence, feature]

prediction = model(x)
prediction = prediction.cpu().detach().numpy()

# prepare plots

plot_range = 10
to_plot_data_y_val = np.zeros(plot_range)
to_plot_data_y_val_pred = np.zeros(plot_range)
to_plot_data_y_test_pred = np.zeros(plot_range)

to_plot_data_y_val[:plot_range-1] = n_btc_scaler.inverse_transform(data_y_val)[-plot_range+1:]
to_plot_data_y_val_pred[:plot_range-1] = n_btc_scaler.inverse_transform(predicted_val)[-plot_range+1:]

to_plot_data_y_test_pred[plot_range-1] = n_btc_scaler.inverse_transform(prediction)[0]

to_plot_data_y_val = np.where(to_plot_data_y_val == 0, None, to_plot_data_y_val)
to_plot_data_y_val_pred = np.where(to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred)
to_plot_data_y_test_pred = np.where(to_plot_data_y_test_pred == 0, None, to_plot_data_y_test_pred)

# plot

plot_date_test = data_date[-plot_range+1:]
plot_date_test.append("tomorrow")

fig = figure(figsize=(25, 5), dpi=80)
fig.patch.set_facecolor((1.0, 1.0, 1.0))
plt.plot(plot_date_test, to_plot_data_y_val, label="Actual prices", marker=".", markersize=10, color=config["plots"]["color_actual"])
plt.plot(plot_date_test, to_plot_data_y_val_pred, label="Past predicted prices", marker=".", markersize=10, color=config["plots"]["color_pred_val"])
plt.plot(plot_date_test, to_plot_data_y_test_pred, label="Predicted price for next day", marker=".", markersize=20, color=config["plots"]["color_pred_test"])
plt.title("Predicted close price of the next trading day")
# plt.grid(b=None, which='major', axis='y', linestyle='--')
plt.grid(which='major', axis='y', linestyle='--')
plt.legend()
plt.show()

print("Predicted close price of the next trading day:", round(to_plot_data_y_test_pred[plot_range-1], 2))