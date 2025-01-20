"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.10
"""

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler

prediction_days = 60


def split_data_function(data: pd.DataFrame, prediction_days: int):
    x_train, y_train = [], []

    for x in range(prediction_days, len(data)):
        x_train.append(data.iloc[x - prediction_days:x]['Close'].values)
        y_train.append(data.iloc[x]['Close'])

    # Konwersja list na tablice numpy
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Zmiana kształtu na format wymagany przez LSTM
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    return x_train, y_train


def train_model_function(x_train, y_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=32)

    return model


def evaluate_model_function(model, btc_data, interval):
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Pobranie danych testowych
    test_start = dt.datetime(2022, 1, 1)
    test_end = dt.datetime.now()
    test_data = yf.download('BTC-USD', start=test_start, end=test_end, interval=interval)
    test_data.columns = test_data.columns.droplevel(1)

    actual_prices = test_data['Close'].values
    total_dataset = pd.concat((btc_data['Close'], test_data['Close']), axis=0)

    # Przygotowanie danych wejściowych
    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.fit_transform(model_inputs)

    x_test = []
    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x - prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Prognozowanie cen
    prediction_prices = model.predict(x_test)
    prediction_prices = scaler.inverse_transform(prediction_prices)

    # Wyświetlenie wyników
    # print(f"Prediction prices for {btc_data}: {prediction_prices}")

    # Prognoza na następny okres
    real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs) + 1, 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    print(f'BTC price for next {interval} prediction: {prediction}$')
    return None


def split_data(btc_preprocessed_data: pd.DataFrame, btc_preprocessed_data_1w: pd.DataFrame):
    """
    Dzieli dane na sekwencje wejściowe (x_train) i wartości docelowe (y_train) dla modelu LSTM.

    Opis:
    -------
    - Model LSTM wymaga danych w postaci sekwencji czasowych jako wejście.
      Celem tej funkcji jest przekształcenie przetworzonych danych cen BTC (preprocessed_btc_data) w taki sposób,
      aby model mógł uczyć się wzorców z przeszłych danych i przewidywać przyszłe wartości.
    - Funkcja tworzy sekwencje historycznych cen o długości prediction_days jako dane wejściowe (x_train),
      a następnie przypisuje do nich wartość kolejnego dnia jako wartość docelową (y_train).

    Szczegóły:
    -----------
    - x_train: Każdy element to sekwencja historyczna, np. ceny BTC z ostatnich prediction_days dni.
    - y_train: Każdy element to cena BTC w dniu następującym po sekwencji w x_train.

    Przykład:
    ----------
    Jeśli preprocessed_btc_data zawiera ceny [100, 102, 104, 106, 108], a prediction_days = 3:
    - x_train będzie zawierać:
        [[100, 102, 104],
         [102, 104, 106]]
    - y_train będzie zawierać:
        [106, 108]

    Argumenty:
    ----------
    btc_preprocessed_data : pd.DataFrame
        DataFrame zawierający przetworzone dane cen BTC, gdzie kolumna 'Close' zawiera wartości cen zamknięcia.

    Zwraca:
    --------
    Tuple[np.ndarray, np.ndarray]
        - x_train (numpy.ndarray): Dane wejściowe w formacie (n_samples, n_timesteps, 1),
          gdzie n_samples to liczba przykładów, a n_timesteps to liczba dni w sekwencji.
        - y_train (numpy.ndarray): Dane wyjściowe zawierające wartości docelowe.

    Szczegóły implementacji:
    -------------------------
    1. Tworzymy puste listy x_train i y_train.
    2. Pętla iteruje od prediction_days do końca danych:
       - W każdej iteracji pobieramy sekwencję cen z ostatnich prediction_days dni
         i dodajemy ją do x_train.
       - Cenę z następnego dnia dodajemy do y_train.
    3. Zamieniamy listy x_train i y_train na tablice numpy.
    4. Kształt x_train zmieniamy na (n_samples, n_timesteps, 1), co jest wymagane przez LSTM.

    """
    # x_train, y_train = [], []
    #
    # for x in range(prediction_days, len(btc_preprocessed_data)):
    #     # Dodajemy sekwencję z ostatnich 'prediction_days' dni do x_train
    #     x_train.append(btc_preprocessed_data.iloc[x - prediction_days:x]['Close'].values)
    #     # Dodajemy wartość docelową (następny dzień) do y_train
    #     y_train.append(btc_preprocessed_data.iloc[x]['Close'])
    #
    # # Konwersja list na tablice numpy
    # x_train, y_train = np.array(x_train), np.array(y_train)
    #
    # # Zmiana kształtu x_train na (liczba przykładów, liczba dni, 1) – wymagane przez LSTM
    # x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # Zrobiłem oddzielna funkcje na górze żeby kodu nie powtarzać
    (x_train_1d, y_train_1d) = split_data_function(btc_preprocessed_data, prediction_days)
    (x_train_1w, y_train_1w) = split_data_function(btc_preprocessed_data_1w, prediction_days)
    return x_train_1d, y_train_1d, x_train_1w, y_train_1w


def train_model(x_train_1d, y_train_1d, x_train_1w, y_train_1w):
    """
    Trenuje model sekwencyjny LSTM do przewidywania wartości na podstawie danych czasowych.

    Model składa się z trzech warstw LSTM z dropoutem, aby zapobiegać przeuczeniu, oraz jednej warstwy Dense na końcu
    do przewidywania wartości wyjściowej. Model jest optymalizowany przy użyciu optymalizatora Adam
    i funkcji straty mean_squared_error.

    Szczegóły warstw:
    -----------------
    - LSTM (Long Short-Term Memory):
        Warstwa LSTM jest specjalnym rodzajem sieci rekurencyjnej (RNN), która może przechowywać informacje
        przez dłuższy czas. Przyjmuje sekwencję danych jako wejście i jest używana do analizy danych czasowych.

        Parametry:
        - units: Liczba jednostek LSTM (czyli liczba ukrytych neuronów w warstwie).
          Większa liczba jednostek może zwiększyć zdolność modelu do uchwycenia złożonych wzorców, ale także
          zwiększa ryzyko przeuczenia i czas treningu.
        - return_sequences: Określa, czy każda jednostka LSTM ma zwracać pełną sekwencję wyjściową.
          Jeśli True, warstwa zwraca sekwencję (np. o kształcie (batch_size, timesteps, units)),
          jeśli False, zwraca tylko ostatnie wyjście (kształt (batch_size, units)).

    - Dropout:
        Dropout to technika regularizacji, która losowo wyłącza (zeruje) część neuronów podczas treningu,
        aby zapobiegać przeuczeniu. Wartość 0.2 oznacza, że 20% neuronów w warstwie będzie losowo wyłączone.

    - Dense:
        Warstwa Dense (gęsta) jest standardową warstwą sieci neuronowej, w której każdy neuron jest połączony
        z każdym neuronem w poprzedniej warstwie. W tym przypadku units=1 oznacza, że warstwa zwraca pojedynczą
        wartość wyjściową, co jest odpowiednie dla problemu regresji (przewidywanie pojedynczej wartości liczbowej).

    Proces:
    -------
    1. Model jest kompilowany z optymalizatorem Adam (algorytm adaptacyjnej optymalizacji) i funkcją straty
       mean_squared_error (średni błąd kwadratowy), co jest standardowym wyborem dla problemów regresji.
    2. Model jest trenowany na danych x_train i y_train przez 25 epok z wielkością batcha równą 32.

    Parametry:
    -----------
    x_train : numpy.ndarray
        Dane wejściowe do treningu modelu w formacie (n_samples, n_timesteps, 1), gdzie:
        - n_samples: liczba przykładów w danych treningowych,
        - n_timesteps: liczba kroków czasowych w każdej sekwencji.
    y_train : numpy.ndarray
        Oczekiwane wartości wyjściowe dla danych treningowych.

    Zwraca:
    --------
    model : tensorflow.keras.Sequential
        Wytrenowany model LSTM.

    """
    model_1d = train_model_function(x_train_1d, y_train_1d)
    model_1w = train_model_function(x_train_1w, y_train_1w)

    return model_1d, model_1w


def evaluate_model(model_1d, btc_preprocessed_data, model_1w, btc_preprocessed_data_1w):
    """
       Ocena modelu na podstawie danych testowych oraz prognoza ceny BTC na następny dzień.

       Opis:
       ---------
       - Funkcja wykorzystuje dane testowe, aby ocenić wydajność modelu LSTM.
       - Używa danych rzeczywistych pobranych z Yahoo Finance, łączy je z przetworzonymi danymi
         i tworzy zestaw testowy.
       - Prognozuje również cenę BTC na następny dzień na podstawie modelu.

        Szczegółowe kroki:
        ------------------
        1. Pobranie rzeczywistych danych testowych z Yahoo Finance od 2022 roku.
        2. Połączenie przetworzonych danych historycznych z nowymi danymi testowymi.
        3. Przygotowanie danych wejściowych dla modelu:
           - Wyodrębnienie sekwencji historycznych (o długości prediction_days).
           - Normalizacja danych w zakresie (0, 1) za pomocą MinMaxScaler.
        4. Przewidywanie cen dla danych testowych.
        5. Prognoza ceny BTC na kolejny dzień na podstawie najnowszych danych historycznych.

       Argumenty:
       ----------
       model : Sequential
           Trenowany model LSTM.
       preprocessed_btc_data : pd.DataFrame
           Przetworzone dane cen BTC zawierające kolumnę 'Close'.

       Zwraca:
       ---------
       None
           Funkcja wypisuje prognozowane ceny oraz cenę BTC na następny dzień w konsoli.

       Przykład działania:
       -------------------
       Funkcja najpierw ocenia model na danych testowych, a następnie przewiduje cenę BTC
       na podstawie sekwencji historycznych.

       """
    # scaler = MinMaxScaler(feature_range=(0, 1))
    #
    # test_start = dt.datetime(2022, 1, 1)
    # test_end = dt.datetime.now()
    # test_data = yf.download('BTC-USD', test_start, test_end)
    # test_data.columns = test_data.columns.droplevel(1)
    #
    # actual_prices = test_data['Close'].values
    # total_dataset = pd.concat((btc_preprocessed_data['Close'], test_data['Close']), axis=0)
    #
    # model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    # model_inputs = model_inputs.reshape(-1, 1)
    # model_inputs = scaler.fit_transform(model_inputs)
    #
    # x_test = []
    # for x in range(prediction_days, len(model_inputs)):
    #     x_test.append(model_inputs[x - prediction_days:x, 0])
    #
    # x_test = np.array(x_test)
    # x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    #
    # prediction_prices = model_1d.predict(x_test)
    # prediction_prices = scaler.inverse_transform(prediction_prices)
    # # print(f'Prediction prices: {prediction_prices}')
    # # Rysowanie plotu
    # # plt.plot(actual_prices, color='black', label='Actual Prices')
    # # plt.plot(prediction_prices, color='green', label='Predicted Prices')
    # # plt.title(f'BTC price prediction')
    # # plt.xlabel('Time')
    # # plt.xlabel('Price')
    # # plt.legend(loc='upper left')
    # # plt.show()
    #
    # # Predict Next Day
    # real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs) + 1, 0]]
    # real_data = np.array(real_data)
    # real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
    #
    # prediction = model_1d.predict(real_data)
    # prediction = scaler.inverse_transform(prediction)
    # print(f'BTC price for next day (1d candles prediction): {prediction}$')

    evaluate_model_function(model_1d, btc_preprocessed_data, "1d")
    evaluate_model_function(model_1w, btc_preprocessed_data_1w, "1wk")

    return None
