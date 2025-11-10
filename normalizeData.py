import pandas as pd
from pandas import DataFrame
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import joblib  # For saving the scaler model

##################################################
# Take the incoming column of the data frame, 
# coerce any non floar to NaN, then fill it forward
#
def ensure_float(df: DataFrame, col_name: str) -> None:
    '''pass in a panda column and ensure it returns with all float '''

    # Convert to numeric, setting non-convertible values to NaN
    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')

    # Forward fill - filling NaNs with the last valid value
    df[col_name].fillna(method='ffill', inplace=True)
    df[col_name] = df[col_name].astype(float)

class Normalizer():
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

##############################################################
    # perform min-max normalization with each column of the data  
    # then persist the scaler 
    # use the same stock scaler for open, adjusted close, high, low
    # it is suggested to do scaler on only the training set, now use the whole set
    #

    # Fit the scaler to the column and transform it

def normalize_data(df: DataFrame)-> DataFrame:
    '''take a set of data and normalize it'''


    #########################################################
    # Need to scale the training data only, and in sections
    # Train the Scaler with training data and smooth data
    # !!! COMMENT THIS OUT for now, worry about smoothing winder later when stuff is working
    #
    # len: int = df.size
    # smoothing_window_size = len /4
    # close_scaler = MinMaxScaler()

    # for di in range(0,len,smoothing_window_size):
    #     close_scaler.fit(df[di:di+smoothing_window_size,:])
    #     df[di:di+smoothing_window_size,:] = close_scaler.transform(df[['adjusted close'],[di:di+smoothing_window_size,:]])

    # # You normalize the last bit of remaining data
    # close_scaler.fit(df[di+smoothing_window_size:,:])
    # df[di+smoothing_window_size:,:] = close_scaler.transform(df[di+smoothing_window_size:,:])

    # normalize
    # note the data are restacked here so must do this every time a new feature is added
    #
    price_scaler = Normalizer()
    n_data_high_price = price_scaler.fit_transform(features_data[:,2])
    n_data_low_price = price_scaler.fit_transform(features_data[:,3])
    n_data_close_price = price_scaler.fit_transform(features_data[:,0])
    volume_scaler = Normalizer()
    n_data_volume = volume_scaler.fit_transform(features_data[:,1])
    spy_scaler = Normalizer()
    n_data_spy_price = spy_scaler.fit_transform(features_data[:,4])
    interest_scaler = Normalizer()
    n_data_interest = interest_scaler.fit_transform(features_data[:,5])
    tenyear_scaler = Normalizer()
    n_data_10year = tenyear_scaler.fit_transform(features_data[:,6])
    vxx_scaler = Normalizer()
    n_data_vxx = vxx_scaler.fit_transform(features_data[:,7])
    unemploy_scaler = Normalizer()
    n_data_unemploy = unemploy_scaler.fit_transform(features_data[:,8])

    ###

    close_scaler = MinMaxScaler()
    df['adjusted close'].to_csv('stock_prenormalized.csv')
    # !!! DEBUG do not normalize close price and see what happens
    df['adjusted close'] = close_scaler.fit_transform(df[['adjusted close']])
    joblib.dump(close_scaler, 'close_scaler.joblib')

    test_scaler = joblib.load('close_scaler.joblib')
    test_df: DataFrame = df['adjusted close']
    test_df = df['adjusted close'].values.reshape(-1, 1)
    test_denorm = test_scaler.inverse_transform(test_df)
    test_df:DataFrame = pd.DataFrame(test_denorm)
    test_df.to_csv('stock_test_denorm.csv')
    

    open_scaler = MinMaxScaler()
    df['open'] = open_scaler.fit_transform(df[['open']])
    joblib.dump(open_scaler, 'open_scaler.joblib')

    high_scaler = MinMaxScaler()
    df['high'] = high_scaler.fit_transform(df[['high']])
    joblib.dump(high_scaler, 'high_scaler.joblib')

    low_scaler = MinMaxScaler()
    df['low'] = low_scaler.fit_transform(df[['low']])
    joblib.dump(low_scaler, 'low_scaler.joblib')    

    volume_scaler = MinMaxScaler()
    # Fit the scaler to the column and transform it
    df['volume'] = volume_scaler.fit_transform(df[['volume']])
    # Saving the scaler for later use
    joblib.dump(volume_scaler, 'volume_scaler.joblib')

    interest_scaler = MinMaxScaler()
    # Fit the scaler to the column and transform it
    df['interest'] = interest_scaler.fit_transform(df[['interest']])
    # Saving the scaler for later use
    joblib.dump(interest_scaler, 'interest_scaler.joblib')

    SPY_scaler = MinMaxScaler()
    # Fit the scaler to the column and transform it
    df['SPY_close'] = SPY_scaler.fit_transform(df[['SPY_close']])
    # Saving the scaler for later use
    joblib.dump(SPY_scaler, 'SPY_scaler.joblib')

    tenyear_scaler = MinMaxScaler()     # can not have a number to lead a var, need to fix latersur
    ensure_float(df, '10year')  # ensure any non float is converted
    # Fit the scaler to the column and transform it
    df['10year']  = tenyear_scaler.fit_transform(df[['10year']])
    # Saving the scaler for later use
    joblib.dump(tenyear_scaler, 'tenyear_scaler.joblib')

    nextday_gain_scaler = MinMaxScaler()
    # Fit the scaler to the column and transform it
    df['nextday_gain'] = nextday_gain_scaler.fit_transform(df[['nextday_gain']])
    # Saving the scaler for later use
    joblib.dump(nextday_gain_scaler, 'nextday_gain_scaler.joblib')

    df.to_csv('TMP_post_etl.csv', index=False) # we want to save the date index
    return df