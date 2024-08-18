import requests
import pandas as pd


def get_all_fear_greed_index():
    url = "https://api.alternative.me/fng/?limit=0&format=json"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()['data']
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')  # Konwersja na format daty
        df = df.rename(columns={'timestamp': 'date', 'value': 'fear_greed_index'})
        df['fear_greed_index'] = df['fear_greed_index'].astype(int)  # Konwersja indeksu na integer
        df.set_index('date', inplace=True)
        return df[['fear_greed_index']]
    else:
        raise Exception("Failed to fetch Fear and Greed Index")


# Przykład użycia
fear_greed_df = get_all_fear_greed_index()
print(fear_greed_df.tail())