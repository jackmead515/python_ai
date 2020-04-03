import matplotlib.pyplot as plot
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras import models
from keras import layers
from keras import optimizers
from keras import losses

def load_confirmed():
  dataset = []
  df = pd.read_csv('deaths.csv')
  df = df.drop(['Province/State', 'Country/Region', 'Lat', 'Long'], axis=1)
  df = df.dropna()
  for index, col in tqdm(enumerate(df.columns[:])):
    spt = col.split('/')
    spt[0] = "0{m}".format(m=spt[0]) if len(spt[0]) == 1 else spt[0]
    spt[1] = "0{d}".format(d=spt[1]) if len(spt[1]) == 1 else spt[1]
    spt[2] = "20{y}".format(y=spt[2])
    t = "{m}-{d}-{y}".format(m=spt[0], d=spt[1], y=spt[2])
    t = datetime.strptime(t, "%m-%d-%Y").timestamp()
    c = 0
    for i, row in df.iterrows():
      if row[index] > 0:
        c += row[index]
    dataset.append([t, c])
  df = pd.DataFrame(dataset)
  df.columns = ['time', 'cases']
  return df

def forecast(data, window_size, future_window):
  datax, datay = [], []
  for i in range(len(data)-window_size-future_window-1):
    datax.append(data[i:(i+window_size), 1])
    datay.append([])
    for x in range(future_window):
      datay[i].append(data[i + x + window_size, 1])
  return datax, datay

def forecast_format_data(dataset, window_size, future_window, train_test_split):
  scaler = MinMaxScaler(feature_range=(0, 1))
  d = np.array(dataset)
  d = scaler.fit_transform(d)
  dx, dy = forecast(d, window_size, future_window)
  tts = int(len(dx)*train_test_split)

  x_train = np.array(dx[:tts]).astype(np.float32)
  x_train = np.expand_dims(x_train, axis=2)

  x_test = np.array(dx[tts+1:]).astype(np.float32)
  x_test = np.expand_dims(x_test, axis=2)

  y_train = np.array(dy[:tts]).astype(np.float32)
  y_train = np.expand_dims(y_train, axis=2)

  y_test = np.array(dy[tts+1:]).astype(np.float32)
  y_test = np.expand_dims(y_test, axis=2)
  
  return x_train, y_train, x_test, y_test, scaler, tts

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

def test_network(model, x_test, y_test, scaler):
  py_test = model.predict(x_test)
  red_color = [1.0,0.0,0.0]
  gre_color = [0.0,1.0,0.0]

  tscaler = MinMaxScaler()
  tscaler.min_, tscaler.scale_ = scaler.min_[1], scaler.scale_[1]
  py_test = tscaler.inverse_transform(py_test)
  y_test = tscaler.inverse_transform(y_test)
  score = np.sqrt(mean_squared_error(py_test, y_test))
  print("Testing Score (RMSE): {}".format(score))
  
  sindex = 0
  for index in range(py_test.shape[0]):
    plot.plot(
      np.arange(sindex, py_test.shape[1]+sindex),
      py_test[index, :],
      color=gre_color,
      marker='.',
      label='predicted'
    )
    sindex += 1
  
  plot.plot(
    np.arange(y_test.shape[1]),
    y_test[0, :],
    color=red_color,
    marker='.',
    label='actual'
  )

  plot.title("Predicted Growth Rate of COVID-19 Cases")
  # plot.legend(loc="upper left")
  plot.xlabel("Days Into Future")
  plot.ylabel("COVID19 Cases")

  plot.show()

if __name__ == "__main__":
  FEATURES = 1
  WINDOW_SIZE = 7
  FUTURE_WINDOW = 30
  TRAIN_TEST_SPLIT = 0.8
  NODES = 32
  EPOCHS = 100
  BATCH_SIZE = 10
  OPTIMIZER = optimizers.Adam()
  LOSS = losses.mean_squared_error

  dataset = load_confirmed()

  print(dataset.shape)
  print(dataset.head())

  x_train, y_train, x_test, y_test, tscaler, tts = forecast_format_data(dataset, WINDOW_SIZE, FUTURE_WINDOW, TRAIN_TEST_SPLIT)

  y_test = np.squeeze(y_test, axis=2)
  y_train = np.squeeze(y_train, axis=2)

  print('tts', tts)
  print('x_train', x_train.shape)
  print('y_train', y_train.shape)
  print('x_test', x_test.shape)
  print('y_test', y_test.shape)

  model = build_network(NODES, WINDOW_SIZE, FUTURE_WINDOW, FEATURES, OPTIMIZER, LOSS)

  train_network(model, x_train, y_train, EPOCHS, BATCH_SIZE)
  test_network(model, x_test, y_test, tscaler)