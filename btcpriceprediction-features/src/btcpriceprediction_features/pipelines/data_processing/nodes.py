"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.10
"""
import pandas as pd
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


def download_data() -> pd.DataFrame:
    """
        Pobiera historyczne dane dotyczące Bitcoina (BTC) z Yahoo Finance.

        Funkcja pobiera dane o cenach Bitcoina w dolarach amerykańskich (USD)
        od 1 stycznia 2016 roku do bieżącej daty, korzystając z API Yahoo Finance.
        Zwracane dane zawierają
        TODO: Ustawić parametry żeby nie było sztywnej daty

        Returns:
            pd.DataFrame: DataFrame zawierający dane o cenach BTC z datą jako indeksem
                          i kolumny dotyczące otwarcia (Open), najwyższej ceny (High),
                          najniższej ceny (Low), zamknięcia (Close), zamknięcia skorygowanego (Adj Close)
                           oraz wolumenu (Volume).
        """

    crypto_currency = 'BTC'
    against_currency = 'USD'
    start = dt.datetime(2016, 1, 1)
    end = dt.datetime.now()
    ticker = f'{crypto_currency}-{against_currency}'
    btc_raw_data = yf.download(ticker, start, end)
    print("Downloading btc data from yahoo")
    print(f"Newest BTC prices:\n {btc_raw_data.tail()}")
    return btc_raw_data


def preprocess_btc_raw(btc_raw_data: pd.DataFrame) -> pd.DataFrame:
    """
        Funkcja przetwarzająca dane BTC, normalizując kolumnę 'Close' do zakresu [0, 1].

        Args:
            btc_raw_data (pd.DataFrame): DataFrame zawierający surowe dane BTC,
                                         w tym kolumnę 'Close' z cenami zamknięcia.

        Returns:
            pd.DataFrame: DataFrame znormalizowany w kolumnie 'Close', zachowujący
                          oryginalne dane w innych kolumnach.
        """
    # Kopiowanie danych, aby nie zmieniać oryginalnego DataFrame
    btc_preprocessed_data = btc_raw_data

    # Inicjalizacja skalera MinMaxScaler do normalizacji danych
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Normalizacja wartości w kolumnie 'Close'
    btc_preprocessed_data['Close'] = scaler.fit_transform(btc_raw_data['Close'].values.reshape(-1, 1))

    return btc_preprocessed_data
