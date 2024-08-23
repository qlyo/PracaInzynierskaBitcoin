import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import wandb
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from autogluon.tabular import TabularPredictor


# Inicjalizacja projektu W&B
wandb.init(project="BTC_Price_Prediction")

prediction_days = 60
crypto_currency = 'BTC'
against_currency = 'USD'
ticker = f'{crypto_currency}-{against_currency}'
scaler = MinMaxScaler(feature_range=(0, 1))


def split_data(data: pd.DataFrame):
    X_train, y_train = [], []
    for x in range(prediction_days, len(data)):
        X_train.append(data.iloc[x - prediction_days:x, data.columns.get_loc('Close')])
        y_train.append(data.iloc[x, data.columns.get_loc('Close')])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    return X_train, y_train, data


def build_model(hp):
    model = Sequential()

    # Definicja przestrzeni hiperparametrów
    model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32),
                   return_sequences=True, input_shape=(prediction_days, 1)))
    model.add(Dropout(hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)))

    model.add(LSTM(units=hp.Int('units_2', min_value=32, max_value=512, step=32), return_sequences=True))
    model.add(Dropout(hp.Float('dropout_rate_2', min_value=0.2, max_value=0.5, step=0.1)))

    model.add(LSTM(units=hp.Int('units_3', min_value=32, max_value=512, step=32)))
    model.add(Dropout(hp.Float('dropout_rate_3', min_value=0.2, max_value=0.5, step=0.1)))

    model.add(Dense(units=1))

    model.compile(optimizer=tf.keras.optimizers.Adam(
        hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
        loss='mean_squared_error')

    return model


def train_model(X_train, y_train):
    # tuner = kt.Hyperband(
    #     build_model,
    #     objective='val_loss',
    #     max_epochs=5,
    #     factor=4,
    #     directory='tuning_results',
    #     project_name='BTC_Price_Prediction',
    #     max_trials=10
    # )
    tuner = kt.RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=10,  # Maksymalna liczba prób (trials)
        executions_per_trial=1,  # Liczba uruchomień na próbę
        directory='tuning_results',
        project_name='BTC_Price_Prediction'
    )

    tuner.search(X_train, y_train, epochs=20, validation_split=0.2)

    # Wybierz najlepszy model po tuningu hiperparametrów
    best_model = tuner.get_best_models(num_models=1)[0]

    # Logowanie metryk treningowych
    history = best_model.fit(X_train, y_train, epochs=25, batch_size=32)
    wandb.log({"loss": history.history['loss'][-1]})

    return best_model


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
    # Plots
    # plt.plot(actual_prices, color='black', label='Actual Prices')
    # plt.plot(prediction_prices, color='green', label='Predicted Prices')
    # plt.title(f'{crypto_currency} price prediction')
    # plt.xlabel('Time')
    # plt.ylabel('Price')
    # plt.legend(loc='upper left')
    # plt.show()
    #
    # Przewidywanie ceny na następny dzień
    real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs) + 1, 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    print("Prediction for next day: $", prediction)


# AutoGluon Node
def train_with_autogluon(X_train: np.ndarray, y_train: np.ndarray):
    # Prepare data for AutoGluon
    X_train = X_train.reshape(X_train.shape[0], -1)
    train_data = pd.DataFrame(X_train)
    train_data['label'] = y_train

    # Train with AutoGluon
    predictor = TabularPredictor(label='label', path='AutogluonModels/').fit(train_data)
    # Logowanie metryk treningowych

    # Logowanie wyników do W&B
    leaderboard = predictor.leaderboard(train_data, silent=True)
    wandb.log({"accuracy": leaderboard.loc[0, 'score_val'],
               "best_model": leaderboard.loc[0, 'model']})
    wandb.log({"leaderboard": wandb.Table(dataframe=leaderboard)})
    return predictor


# Evaluate Node
def evaluate_autogluon_model(predictor, data):
    # Prepare the test data similarly
    prediction_days = 60
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
    x_test = x_test.reshape(x_test.shape[0], -1)

    prediction_prices = predictor.predict(pd.DataFrame(x_test))

    # Logowanie przewidywań do W&B
    wandb.log({"predictions": wandb.Table(dataframe=pd.DataFrame(prediction_prices, columns=['Predicted Prices']))})

    return prediction_prices

