"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.6
"""
import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler


def preprocess_btc_raw(btc_raw: pd.DataFrame) -> pd.DataFrame:
    # Load the data

    btc_raw = btc_raw.drop(columns=['next_day_close'])
    print(btc_raw.head())
    return btc_raw
