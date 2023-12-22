import yfinance as yf
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import set_random_seed
import math
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

tf.get_logger().setLevel("WARNING")
SEED = 69


def read_data(start_date, end_date, symbol="META"):
    ticker = yf.Ticker(symbol)
    return ticker.history(interval="1d", start=start_date, end=end_date)


def create_model_simpler(
    units,
    period=15,
):
    set_random_seed(SEED)
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=(period, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=units, input_shape=(period, 1)))
    model.add(Dense(units=32, activation="relu"))
    model.add(Dense(units=1))
    model.compile(optimizer="adam", loss="mse")
    return model


def create_model_original(units, period=15):
    set_random_seed(SEED)
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=(period, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=units, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=units, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=units))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def get_stats(y_true, y_pred):
    MAE = mean_absolute_error(y_true, y_pred)
    MAPE = mean_absolute_percentage_error(y_true, y_pred)
    RMSE = math.sqrt(mean_squared_error(y_true, y_pred))
    print(f"MAE = {MAE}")
    print(f"MAPE = {MAPE}")
    print(f"RMSE = {RMSE}")
    return MAE, MAPE, RMSE


def create_dataset(dataset, n_timesteps=1):
    X, y = [], []
    for i in range(len(dataset) - n_timesteps):
        v = dataset[i : (i + n_timesteps), 0]
        X.append(v)
        y.append(dataset[i + n_timesteps, 0])
    return np.array(X), np.array(y)
