import datetime as dt
from pathlib import Path
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from src.btc_price_prediction.pipelines.data_processing.nodes import download_data, preprocess_btc_raw

# Uzyskanie ścieżki do katalogu głównego projektu
project_path = Path(__file__).resolve().parent

# Inicjalizacja projektu Kedro
bootstrap_project(project_path)

# Tworzenie sesji Kedro
with KedroSession.create(project_path=".") as session:
    context = session.load_context()
    catalog = context.catalog

# Krok 3: Debugowanie funkcji `download_data`
print("=== Testowanie download_data ===")
crypto_currency = "BTC"
against_currency = "USD"
start_date = dt.datetime(2021, 1, 1)
end_date = "now"
force_download = True

try:
    btc_raw_data = download_data(crypto_currency, against_currency, start_date, end_date, force_download)
    print("Pobrano dane BTC:")
    print(btc_raw_data.head())
except Exception as e:
    print(f"Błąd w download_data: {e}")
    btc_raw_data = None  # Zabezpieczenie przed brakiem danych

# Krok 4: Debugowanie funkcji `preprocess_btc_raw`
if btc_raw_data is not None:
    try:
        btc_preprocessed_data = preprocess_btc_raw(btc_raw_data)
        print("Przetworzone dane BTC:")
        print(btc_preprocessed_data.head())
    except Exception as e:
        print(f"Błąd w preprocess_btc_raw: {e}")
else:
    print("Nie można przetestować preprocess_btc_raw z powodu braku danych wejściowych.")
