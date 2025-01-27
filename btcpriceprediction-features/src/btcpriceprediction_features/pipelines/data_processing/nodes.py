"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.10
"""
from typing import Tuple, Any

import pandas as pd
import datetime as dt
import yfinance as yf
from pandas import DataFrame, Series
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def download_data():
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
    train_dates = pd.to_datetime(btc_raw_dataset_1d.index)
    print(btc_raw_dataset_1d.index)
    return btc_raw_dataset_1d, btc_raw_dataset_1w, train_dates


def preprocess_btc_raw(btc_raw_dataset_1d: pd.DataFrame, btc_raw_dataset_1w: pd.DataFrame) -> tuple[
    DataFrame, DataFrame, MinMaxScaler, MinMaxScaler]:
    """
    Funkcja przetwarzająca dane BTC, normalizując kolumny do zakresu [0, 1].

    Args:
        btc_raw_dataset_1d (pd.DataFrame): DataFrame zawierający surowe dane BTC (dzienny interwał).
        btc_raw_dataset_1w (pd.DataFrame): DataFrame zawierający surowe dane BTC (tygodniowy interwał).

    Returns:
        tuple: DataFrame znormalizowanych danych dziennych, tygodniowych oraz obiekt MinMaxScaler.
    """

    # Inicjalizacja skalera MinMaxScaler do normalizacji danych
    scaler1d = MinMaxScaler()
    scaler1w = MinMaxScaler()
    print(btc_raw_dataset_1d)
    # Wyodrębnienie kolumn do normalizacji
    cols_to_normalize = list(btc_raw_dataset_1d.columns[2:7])

    # Normalizacja danych dziennych
    btc_preprocessed_data_1d = btc_raw_dataset_1d.copy()
    btc_preprocessed_data_1d = scaler1d.fit_transform(btc_raw_dataset_1d[cols_to_normalize].astype(float))

    # Normalizacja danych tygodniowych
    btc_preprocessed_data_1w = btc_raw_dataset_1w.copy()
    btc_preprocessed_data_1w = scaler1w.fit_transform(btc_raw_dataset_1w[cols_to_normalize].astype(float))

    print(f"Dane z btc_raw_1d na koniec skalowania:\n {btc_preprocessed_data_1d}")
    print(f"Dane z btc_raw_1w na koniec skalowania\n {btc_preprocessed_data_1w}")

    return btc_preprocessed_data_1d, btc_preprocessed_data_1w, scaler1d,scaler1w
