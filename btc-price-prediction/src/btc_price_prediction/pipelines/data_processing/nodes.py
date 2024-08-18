"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.6
"""
import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from kedro.framework.session import KedroSession
from pathlib import Path


def download_data(crypto_currency: str, against_currency: str, start_date: dt.datetime, end_date: str,
                  force_download: bool) -> pd.DataFrame:
    project_path = Path.cwd()

    # Inicjalizacja sesji Kedro i katalogu danych
    with KedroSession.create(project_path=project_path) as session:
        context = session.load_context()
        catalog = context.catalog

        if end_date == "now":
            end_date = dt.datetime.now()
        print('*'*20)
        if not force_download:
            try:
                btc_raw_data = catalog.load("btc_raw_data")
                print('Data loaded from catalog')
            except:
                print('Data not found in catalog, downloading data')
                ticker = f'{crypto_currency}-{against_currency}'
                btc_raw_data = yf.download(ticker, start=start_date, end=end_date)
                catalog.save("btc_raw_data", btc_raw_data)
        else:
            print("Force download data")
            ticker = f'{crypto_currency}-{against_currency}'
            btc_raw_data = yf.download(ticker, start=start_date, end=end_date)
            catalog.save("btc_raw_data", btc_raw_data)
        print('*' * 20)
    return btc_raw_data


def preprocess_btc_raw(btc_raw: pd.DataFrame) -> pd.DataFrame:
    # pobieranie z yf
    crypto_currency = 'BTC'
    against_currency = 'USD'
    start = dt.datetime(2016, 1, 1)
    end = dt.datetime.now()

    # Pobieranie danych za pomocą yfinance
    ticker = f'{crypto_currency}-{against_currency}'
    btc_preprocessed_data = yf.download(ticker, start=start, end=end)

    #################################################################################
    # używanie z kegla danych
    # Dropping columns
    # btc_preprocessed_data = btc_raw.drop(columns=['next_day_close'])
    #################################################################################
    # Scaling values
    scaler = MinMaxScaler(feature_range=(0, 1))
    btc_preprocessed_data['Close'] = scaler.fit_transform(btc_preprocessed_data['Close'].values.reshape(-1, 1))
    print(btc_preprocessed_data.head())
    return btc_preprocessed_data
