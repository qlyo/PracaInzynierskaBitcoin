"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.6
"""
import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

def preprocess_btc_raw(btc_raw: pd.DataFrame) -> pd.DataFrame:
    #pobieranie z yf
    crypto_currency = 'BTC'
    against_currency = 'USD'
    start = dt.datetime(2016, 1, 1)
    end = dt.datetime.now()

    # Pobieranie danych za pomocą yfinance
    ticker = f'{crypto_currency}-{against_currency}'
    btc_preprocessed_data = yf.download(ticker, start=start, end=end)

    #################################################################################
    #używanie z kegla danych
    # Dropping columns
    #btc_preprocessed_data = btc_raw.drop(columns=['next_day_close'])
    #################################################################################
    # Scaling values
    scaler = MinMaxScaler(feature_range=(0, 1))
    btc_preprocessed_data['Close'] = scaler.fit_transform(btc_preprocessed_data['Close'].values.reshape(-1, 1))
    print(btc_preprocessed_data.head())
    return btc_preprocessed_data
