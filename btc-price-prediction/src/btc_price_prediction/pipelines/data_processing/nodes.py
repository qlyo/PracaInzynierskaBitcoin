"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.6
"""
import pandas as pd
import datetime as dt
import requests
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from kedro.framework.session import KedroSession
from pathlib import Path



def get_all_fear_greed_index():
    url = "https://api.alternative.me/fng/?limit=0&format=json"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()['data']
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='s')  # Konwersja na format daty
        df = df.rename(columns={'timestamp': 'date', 'value': 'fear_greed_index'})
        df['fear_greed_index'] = df['fear_greed_index'].astype(int)  # Konwersja indeksu na integer
        df.set_index('date', inplace=True)
        return df[['fear_greed_index']]
    else:
        raise Exception("Failed to fetch Fear and Greed Index")


def download_data(crypto_currency: str, against_currency: str, start_date: dt.datetime, end_date: str,
                  force_download: bool) -> pd.DataFrame:
    project_path = Path.cwd()

    # Pobieranie datasetu BTC
    with KedroSession.create(project_path=project_path) as session:
        context = session.load_context()
        catalog = context.catalog

        if end_date == "now":
            end_date = dt.datetime.now()

        # Pobierz dane Fear and Greed Index
        fear_greed_df = get_all_fear_greed_index()

        # Ustal najwcześniejszą datę w danych Fear and Greed Index
        earliest_fng_date = fear_greed_df.index.min()
        print(f"Earliest Fear and Greed Index date: {earliest_fng_date}")

        # Upewnij się, że BTC dane są pobierane od najwcześniejszej dostępnej daty w indeksie
        start_date = pd.to_datetime(start_date)
        if start_date < earliest_fng_date:
            start_date = earliest_fng_date

        print('*' * 20)
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
        print('*' * 20)

        # Połącz BTC dane z Fear and Greed Index
        if 'fear_greed_index' not in btc_raw_data.columns:
            btc_raw_data = btc_raw_data.merge(fear_greed_df, how='left', left_index=True, right_index=True)
            btc_raw_data.fillna(method='ffill', inplace=True)  # Uzupełnij brakujące wartości
        catalog.save("btc_raw_data", btc_raw_data)
        print(btc_raw_data.index)

    return btc_raw_data


def preprocess_btc_raw(btc_raw: pd.DataFrame) -> pd.DataFrame:
    btc_preprocessed_data = btc_raw
    # Scaling values
    scaler = MinMaxScaler(feature_range=(0, 1))
    btc_preprocessed_data['Close'] = scaler.fit_transform(btc_preprocessed_data['Close'].values.reshape(-1, 1))
    print("Last records of dataset: ")
    print(btc_preprocessed_data.tail())
    return btc_preprocessed_data
