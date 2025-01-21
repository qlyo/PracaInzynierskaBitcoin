"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.10
"""
from typing import Tuple

import pandas as pd
import datetime as dt
import yfinance as yf
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler


def download_data() -> tuple[DataFrame | None, DataFrame | None]:
    """
        Pobiera historyczne dane dotyczące Bitcoina (BTC) z Yahoo Finance.

        Funkcja pobiera dane o cenach Bitcoina w dolarach amerykańskich (USD)
        od 1 stycznia 2016 roku do bieżącej daty, korzystając z API Yahoo Finance.
        Zwracane dane zawierają
        TODO: Ustawić parametry żeby nie było sztywnej daty
        TODO: przestać tyrać
        Returns:
            pd.DataFrame: DataFrame zawierający dane o cenach BTC z datą jako indeksem
                          i kolumny dotyczące otwarcia (Open), najwyższej ceny (High),
                          najniższej ceny (Low), zamknięcia (Close), zamknięcia skorygowanego (Adj Close)
                           oraz wolumenu (Volume).
        """

    crypto_currency = 'BTC'
    against_currency = 'USD'
    start = dt.datetime(2018, 1, 1)
    end = dt.datetime.now()
    ticker = f'{crypto_currency}-{against_currency}'
    btc_raw_dataset_1d = yf.download(ticker, start, end)
    btc_raw_dataset_1w = yf.download(ticker, start, end, interval="1wk")
    # Usuwanie wiersza z ticker BTC-USD (error z multi kolumnami
    btc_raw_dataset_1d.columns = btc_raw_dataset_1d.columns.droplevel(1)
    btc_raw_dataset_1w.columns = btc_raw_dataset_1w.columns.droplevel(1)

    print("Downloading btc data from yahoo")
    print(f"Newest BTC prices in dataset:\n {btc_raw_dataset_1d.tail()}")
    print(f"Oldest BTC prices in dataset:\n {btc_raw_dataset_1d.head()}")

    print(f"BTC 1w candles:\n {btc_raw_dataset_1w}")
    return btc_raw_dataset_1d, btc_raw_dataset_1w


def preprocess_btc_raw(btc_raw_dataset_1d: pd.DataFrame, btc_raw_dataset_1w: pd.DataFrame) -> tuple[
    DataFrame, DataFrame]:
    """
        Funkcja przetwarzająca dane BTC, normalizując kolumnę 'Close' do zakresu [0, 1].

        Args:
            btc_raw_dataset_1w:
            btc_raw_dataset_1d (pd.DataFrame): DataFrame zawierający surowe dane BTC,
                                         w tym kolumnę 'Close' z cenami zamknięcia.

        Returns:
            pd.DataFrame: DataFrame znormalizowany w kolumnie 'Close', zachowujący
                          oryginalne dane w innych kolumnach.
        """

    # Kopiowanie danych, aby nie zmieniać oryginalnego DataFrame
    btc_preprocessed_data_1d = btc_raw_dataset_1d
    btc_preprocessed_data_1w = btc_raw_dataset_1w

    # Inicjalizacja skalera MinMaxScaler do normalizacji danych
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Normalizacja wartości w kolumnie 'Close'
    btc_preprocessed_data_1d['Close'] = scaler.fit_transform(btc_raw_dataset_1d['Close'].values.reshape(-1, 1))
    btc_preprocessed_data_1w['Close'] = scaler.fit_transform(btc_raw_dataset_1w['Close'].values.reshape(-1, 1))
    print(f"Zeskalowane dane z btc_raw:\n {btc_preprocessed_data_1d}")
    print(f"Zeskalowane dane z btc_raw_1w:\n {btc_preprocessed_data_1w}")

    return btc_preprocessed_data_1d, btc_preprocessed_data_1w
