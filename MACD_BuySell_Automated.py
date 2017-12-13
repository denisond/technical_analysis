import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import datetime
import math


pd.options.mode.chained_assignment = None

def symbol_to_path(symbol, base_dir="data"):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))

def get_data(symbol, dates):
    """Read in daily price(adjusted close) of asset from CSV files for a given set of dates."""
    df = pd.DataFrame(index=dates)
    df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date', parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])# Read in entire data file.
    df = df.join(df_temp, how='inner') # Merge data-frames, setting specified data range as index.
    df['Adj Close'].fillna(method='ffill', inplace=True)# Make sure data is nice and clean with no missing values.
    return df


def get_rolling_mean(values, window):
    return values.rolling(window = window).mean()

def get_rolling_std(values, window):
    return values.rolling(window = window).std()

def get_EMA(values, span):
    return  values.ewm(span=span).mean()

def get_upperband(rm,std):
    upper_band = rm + std*2
    return upper_band

def get_lowerband(rm,std):
    lower_band = rm - std*2
    return lower_band

def plot_BB_and_RM_and_MACD(df, title):
    """Plots asset price, bollinger bands, rolling mean, moving average convergence divergence, and signal line."""
    fig,ax1 = plt.subplots(1,1)
    ax1.plot(df[['Adj Close','RollingMean','UpperBand','LowerBand']])
    y = ax1.get_ylim()
    ax1.set_ylim(y[0] - (y[1]-y[0])*0.4, y[1])

    ax2 = ax1.twinx()
    ax2.set_position(matplotlib.transforms.Bbox([[0.125,0.1],[0.9,0.32]]))
    ax2.plot(df['SignalLine'], color='#77dd77')
    ax2.plot(df['MACD'], color='#dd4444')
    ax1.set_title(title)
    ax1.legend(['Adj Close','RollingMean','UpperBand','LowerBand'], loc='upper left')
    ax2.legend(['SignalLine','MACD'], loc='lower left')
    plt.show()

def test_run():
    dates = pd.date_range('2012-01-01', '2012-09-12')
    symbol = 'XOM'
    data = get_data(symbol, dates)

    data['RollingMean'] = get_rolling_mean(data['Adj Close'], 30)
    data['UpperBand'] = get_upperband(data['RollingMean'], get_rolling_std(data['Adj Close'], 30))
    data['LowerBand'] = get_lowerband(data['RollingMean'], get_rolling_std(data['Adj Close'], 30))
    data['12 EMA']= get_EMA(data['Adj Close'], span= 12)
    data['26 EMA'] = get_EMA(data['Adj Close'], span=26)
    data['MACD']=(data['12 EMA']-data['26 EMA'])
    data['SignalLine']=get_rolling_mean(data['MACD'], 7)#try EMA as well

    data['Buy'] = 0
    data['Sell'] = 0
    data['Gains'] = 0
    data['StopBuy'] = 0
    data['StopSell'] = 0

    for i in range(0, len(data)):
        if math.isclose(data['Adj Close'].iloc[i], data['RollingMean'].iloc[i], rel_tol= .03):
            data['StopBuy'].iloc[i] = 0
            data['StopSell'].iloc[i] = 0
            data['Gains'].iloc[i] = abs(data['Adj Close'].iloc[i] - data['Adj Close'].iloc[i-1])

        if data['MACD'].iloc[i] > data['SignalLine'].iloc[i] and data['StopBuy'].iloc[i-1] == 0:
            data['Buy'].iloc[i] = 1
            data['Gains'].iloc[i] = data['Adj Close'].iloc[i] - data['Adj Close'].iloc[i - 1]


        if data['MACD'].iloc[i] < data['SignalLine'].iloc[i] and data['StopSell'].iloc[i-1] == 0:
            data['Sell'].iloc[i] = -1
            data['Gains'].iloc[i] = -(data['Adj Close'].iloc[i] - data['Adj Close'].iloc[i-1])


        if data['MACD'].iloc[i] > data['SignalLine'].iloc[i] and math.isclose(data['UpperBand'].iloc[i],data['Adj Close'].iloc[i],rel_tol= .02):
            data['StopBuy'].iloc[i] = 1
            data['Buy'].iloc[i] = 0
            data['Gains'].iloc[i] = (data['Adj Close'].iloc[i] - data['Adj Close'].iloc[i-1])


        if data['MACD'].iloc[i] < data['SignalLine'].iloc[i] and math.isclose(data['LowerBand'].iloc[i], data['Adj Close'].iloc[i],rel_tol= .03):
            data['StopSell'].iloc[i] = 1
            data['Sell'].iloc[i] = 0
            data['Gains'].iloc[i] = -(data['Adj Close'].iloc[i] - data['Adj Close'].iloc[i-1])

    total_gains = data['Gains'].sum()
    plot_BB_and_RM_and_MACD(data, '{} Technical Analysis'.format(symbol))

if __name__ == "__main__":
    test_run()