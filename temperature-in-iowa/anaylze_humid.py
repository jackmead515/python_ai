import matplotlib.pyplot as plot
import pandas as pd
import numpy as np
import math
import time
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from datetime import datetime

import util

def forecast(data, window_size, future_window):
  datax, datay = [], []
  for i in range(len(data)-window_size-future_window-1):
    datax.append(data[i:(i+window_size), 1])
    datay.append([])
    for x in range(future_window):
      datay[i].append(data[i + x + window_size, 1])
  return datax, datay

# def forecast_temp_humid(data, window_size, future_window):
#   datax, datay = [], []
#   for i in range(len(data)-window_size-future_window-1):
#     datax.append([
#       data[i:(i+window_size), 1],
#       data[i:(i+window_size), 2]
#     ])
#     datay.append([])
#     for x in range(future_window):
#       datay[i].append(data[i + x + window_size, 1])
#   return datax, datay

def format_traintest_data(df, window_size, future_window, train_test_split):
  scaler = MinMaxScaler(feature_range=(-1, 1))
  dataset = np.array(df)
  dataset = scaler.fit_transform(dataset)
  datax, datay = forecast(dataset, window_size, future_window)
  tts = int(len(datax)*train_test_split)
  x_train = np.array(datax[:tts]).astype(np.float32)
  y_train = np.array(datay[:tts]).astype(np.float32)
  x_test = np.array(datax[tts+1:]).astype(np.float32)
  y_test = np.array(datay[tts+1:]).astype(np.float32)
  x_train = np.expand_dims(x_train, axis=2)
  y_train = np.expand_dims(y_train, axis=2)
  x_test = np.expand_dims(x_test, axis=2)
  y_test = np.expand_dims(y_test, axis=2)
  return x_train, y_train, x_test, y_test, scaler

def format_validation_data(df, window_size, future_window):
  scaler = MinMaxScaler(feature_range=(-1, 1))
  dataset = np.array(df)
  dataset = scaler.fit_transform(dataset)
  datax, datay = forecast(dataset, window_size, future_window)
  x_test = np.expand_dims(np.array(datax).astype(np.float32), axis=2)
  y_test = np.expand_dims(np.array(datay).astype(np.float32), axis=2)
  return x_test, y_test, scaler

def build_network(nodes, window_size, future_window, features, optimizer, loss):
  model = models.Sequential()
  #dropout=0.2, recurrent_dropout=0.2
  model.add(layers.LSTM(nodes, input_shape=(window_size, features)))
  model.add(layers.Dense(future_window))
  model.compile(optimizer=optimizer, loss=loss)
  return model

def train_network(model, x_train, y_train, epochs, batch_size):
  start_time = time.time()
  history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
  end_time = time.time()
  print('Training Time: ' + str(end_time-start_time) + ' secs')
  plot.plot(np.arange(epochs), history.history['loss'])
  plot.show()

def evaluate_network(model, x_test, y_test, batch_size):
  score = model.evaluate(x_test, y_test)
  print('Score: {}'.format(score))

def test_network(model, x_test, y_test, scaler):
  pred_test = model.predict(x_test)
  tscaler = MinMaxScaler()
  tscaler.min_, tscaler.scale_ = scaler.min_[1], scaler.scale_[1]
  pred_test = tscaler.inverse_transform(pred_test)
  y_test = tscaler.inverse_transform(y_test)
  score = np.sqrt(mean_squared_error(pred_test, y_test))
  print("Testing Score (RMSE): {}".format(score))

  plot.plot(np.arange(pred_test.shape[1]), pred_test[0, :], color=[0.0,0.0,1.0], marker='.')
  # plot.plot(np.arange(y_test.shape[0]), y_test[:, 0])
  plot.plot(np.arange(48), y_test[:48, 0], color=[1.0,0.0,0.0], marker='.')
  plot.show()

if __name__ == "__main__":
  print('Loading data...')
  df = util.load_humid_data()

  FEATURES = 1
  WINDOW_SIZE = 48
  FUTURE_WINDOW = 24
  TRAIN_TEST_SPLIT = 0.8
  NODES = 16
  EPOCHS = 100
  BATCH_SIZE = 500
  OPTIMIZER = optimizers.Adam()
  LOSS = losses.mean_squared_error

  years = util.split_years(df)
  months = util.split_months(years[9])
  x_train, y_train, x_test, y_test, tscaler = format_traintest_data(years[9], WINDOW_SIZE, FUTURE_WINDOW, TRAIN_TEST_SPLIT)

  # x_train = x_train.reshape((x_train.shape[0], x_train.shape[2], x_train.shape[1]))
  # x_test = x_test.reshape((x_test.shape[0], x_test.shape[2], x_test.shape[1]))
  y_test = np.squeeze(y_test, axis=2)
  y_train = np.squeeze(y_train, axis=2)

  print('x_train', x_train.shape)
  print('y_train', y_train.shape)

  #x_valid, y_valid, vscaler = format_validation_data(years[10], WINDOW_SIZE)

  ###################################################################################
  # Model

  print('Running network...')
  model = build_network(NODES, WINDOW_SIZE, FUTURE_WINDOW, FEATURES, OPTIMIZER, LOSS)
  train_network(model, x_train, y_train, EPOCHS, BATCH_SIZE)
  test_network(model, x_test, y_test, tscaler)
  #predict_into_future(model, y_valid, WINDOW_SIZE, vscaler)
