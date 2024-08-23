"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.6
"""

import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt
import wandb
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler


# Inicjalizacja projektu W&B
wandb.init(project="BTC_Price_Prediction")


prediction_days = 60
crypto_currency = 'BTC'
against_currency = 'USD'
ticker = f'{crypto_currency}-{against_currency}'
scaler = MinMaxScaler(feature_range=(0, 1))
def split_data(data: pd.DataFrame):
    prediction_days = 60  # Upewnij się, że masz wartość prediction_days

    X_train, y_train = [], []

    for x in range(prediction_days, len(data)):
        X_train.append(data.iloc[x - prediction_days:x, data.columns.get_loc('Close')])
        y_train.append(data.iloc[x, data.columns.get_loc('Close')])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    return X_train, y_train, data


def train_model(X_train, y_train):

    # Creating Neural Network
    model = Sequential()
    #
    # model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    # model.add(Dropout(0.2))
    # model.add(LSTM(units=50, return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(LSTM(units=50))
    # model.add(Dropout(0.2))
    # model.add(Dense(units=1))
    #
    # model.compile(optimizer='adam', loss='mean_squared_error')
    # Definiowanie przestrzeni hiperparametrów dla liczby jednostek (units)
    model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32),
                   return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)))

    model.add(LSTM(units=hp.Int('units_2', min_value=32, max_value=512, step=32), return_sequences=True))
    model.add(Dropout(hp.Float('dropout_rate_2', min_value=0.2, max_value=0.5, step=0.1)))

    model.add(LSTM(units=hp.Int('units_3', min_value=32, max_value=512, step=32)))
    model.add(Dropout(hp.Float('dropout_rate_3', min_value=0.2, max_value=0.5, step=0.1)))

    model.add(Dense(units=1))

    # Logowanie metryk treningowych
    history = model.fit(X_train, y_train, epochs=25, batch_size=32)
    # Logowanie strat dla każdej epoki do Weights & Biases
    for epoch, loss in enumerate(history.history['loss']):
        wandb.log({"loss": loss})
    return model


def evaluate_model(model, data):
    test_start = dt.datetime(2020, 1, 1)
    test_end = dt.datetime.now()

    test_data = yf.download(ticker, start=test_start, end=test_end)
    actual_prices = test_data['Close'].values

    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.fit_transform(model_inputs)

    x_test = []

    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x - prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    prediction_prices = model.predict(x_test)
    prediction_prices = scaler.inverse_transform(prediction_prices)


    #Plots
    # plt.plot(actual_prices, color='black', label='Actual Prices')
    # plt.plot(prediction_prices, color='green', label='Predicted Prices')
    # plt.title(f'{crypto_currency} price prediction')
    # plt.xlabel('Time')
    # plt.ylabel('Price')
    # plt.legend(loc='upper left')
    # plt.show()
    #

    #Predict next day
    real_data = [model_inputs[len(model_inputs) +1 - prediction_days:len(model_inputs)+1,0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data,(real_data.shape[0], real_data.shape[1],1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    print("Prediction for next day: $")
    print(prediction)