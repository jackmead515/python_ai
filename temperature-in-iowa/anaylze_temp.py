import matplotlib.pyplot as plot
import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from datetime import datetime

def load_data():
  df = pd.read_csv('ames_iowa_30-09-1996_26-10-2019_temp.csv')
  df = df.drop(['station'], axis=1)
  df = df.dropna()
  df.columns = ['time', 'value']
  # df.time = pd.DatetimeIndex(df.time).astype(np.int64)
  df.time = [datetime.strptime(x, "%Y-%m-%d %H:%M").timestamp() for x in df.time]
  return df

def forecast(data, window_size):
  datax, datay = [], []
  for i in range(len(data)-window_size-1):
    datax.append(data[i:(i+window_size), 1])
    datay.append(data[i + window_size, 1])
  return datax, datay

def split_years(df):
  values = np.array(df)
  years = [[]]
  index = 0
  current_year = datetime.fromtimestamp(values[0, 0]).year
  for pair in values:
    time = pair[0]
    temp = pair[1]
    data_year = datetime.fromtimestamp(time).year
    if data_year == current_year:
      years[index].append(np.array([time, temp]))
    else:
      index += 1
      current_year = data_year
      years.append([])
  return years

def graph_years(df):
  years = np.array(split_years(df))
  figure = plot.figure(facecolor='white')
  plot_index = 1
  for year in years:
    year = np.array(year)
    axis = figure.add_subplot(4, 6, plot_index)
    axis.set_xlabel(datetime.fromtimestamp(year[0, 0]).year)
    plot.xticks([])
    plot.yticks([])
    plot.plot(year[:, 0], year[:, 1])
    plot_index+=1
  plot.show()

def split_months(year):
  values = np.array(year)
  months = [[]]
  index = 0
  current_month = datetime.fromtimestamp(values[0, 0]).month
  for pair in values:
    time = pair[0]
    temp = pair[1]
    data_month = datetime.fromtimestamp(time).month
    if data_month == current_month:
      months[index].append(np.array([time, temp]))
    else:
      index += 1
      current_month = data_month
      months.append([])
  return months

def split_days(month):
  values = np.array(month)
  days = [[]]
  index = 0
  current_day = datetime.fromtimestamp(values[0, 0]).day
  for pair in values:
    time = pair[0]
    temp = pair[1]
    data_day = datetime.fromtimestamp(time).day
    if data_day == current_day:
      days[index].append(np.array([time, temp]))
    else:
      index += 1
      current_day = data_day
      days.append([])
  return days

def graph_months(df):
  years = split_years(df)
  months = np.array(split_months(years[9]))
  figure = plot.figure(facecolor='white')
  plot_index = 1
  for month in months:
    month = np.array(month)
    axis = figure.add_subplot(3, 4, plot_index)
    axis.set_xlabel(datetime.fromtimestamp(month[0, 0]).month)
    plot.xticks([])
    plot.yticks([])
    plot.plot(month[:, 0], month[:, 1])
    plot_index+=1
  plot.show()

def graph_days(df):
  years = split_years(df)
  months = split_months(years[9])
  days = np.array(split_days(months[5]))
  figure = plot.figure(facecolor='white')
  plot_index = 1
  for day in days:
    day = np.array(day)
    axis = figure.add_subplot(6, 5, plot_index)
    axis.set_xlabel(datetime.fromtimestamp(day[0, 0]).day)
    plot.xticks([])
    plot.yticks([])
    plot.plot(day[:, 0], day[:, 1])
    plot_index+=1
  plot.show()

def graph_whole(df):
  df = df.set_index('time')
  axis = df.plot.line(grid=True)
  axis.set_yticklabels(['{}C'.format(y) for y in axis.get_yticks()])
  axis.set_ylabel('Temperature in C')
  axis.set_xticklabels([datetime.utcfromtimestamp(x).strftime('%Y') for x in axis.get_xticks()])
  axis.set_xlabel('Time (1996 - 2019)')
  axis.get_legend().remove()
  plot.show()
  df = df.reset_index()

def points_per_month(df):
  years = split_years(df)
  months = np.array(split_months(years[9]))
  lengths = []
  for i in range(len(months)):
    lengths.append(len(months[i]))
    print('Month: ' + str(i) + ' data points: ' + str(len(months[i])))
  print('Median: ' + str(np.median(lengths)))
  print('Mode: ' + str(np.average(lengths)))

def format_traintest_data(df, window_size, train_test_split):
  scaler = MinMaxScaler(feature_range=(-1, 1))
  dataset = np.array(df)
  dataset = scaler.fit_transform(dataset)
  datax, datay = forecast(dataset, window_size)
  tts = int(len(datax)*train_test_split)
  x_train = np.expand_dims(np.array(datax[:tts]).astype(np.float32), axis=2)
  y_train = np.expand_dims(np.array(datay[:tts]).astype(np.float32), axis=2)
  x_test = np.expand_dims(np.array(datax[tts+1:]).astype(np.float32), axis=2)
  y_test = np.expand_dims(np.array(datay[tts+1:]).astype(np.float32), axis=2)
  return x_train, y_train, x_test, y_test, scaler

def format_validation_data(df, window_size):
  scaler = MinMaxScaler(feature_range=(-1, 1))
  dataset = np.array(df)
  dataset = scaler.fit_transform(dataset)
  datax, datay = forecast(dataset, window_size)
  x_test = np.expand_dims(np.array(datax).astype(np.float32), axis=2)
  y_test = np.expand_dims(np.array(datay).astype(np.float32), axis=2)
  return x_test, y_test, scaler

def build_network(nodes, window_size, features, optimizer, loss):
  model = models.Sequential()
  #dropout=0.2, recurrent_dropout=0.2
  model.add(layers.LSTM(nodes, input_shape=(window_size, features)))
  model.add(layers.Dense(1))
  model.compile(optimizer=optimizer, loss=loss)
  return model

def train_network(model, x_train, y_train, epochs, batch_size):
  history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
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

  # plot.plot(np.arange(pred_test.shape[0]), pred_test[:, 0])
  # plot.plot(np.arange(y_test.shape[0]), y_test[:, 0])
  # plot.show()

def predict_into_future(model, test, window_size, scaler):
  preds = []
  sequence = [test[i, 0] for i in range(window_size)]

  for _ in range(500):
    pd = model.predict(np.array([
      np.expand_dims(np.array(sequence), axis=1)
    ]))[0, 0]
    preds.append(pd)
    sequence.append(pd)
    sequence.pop(0)
  
  preds = np.expand_dims(np.array(preds), axis=1)
  tscaler = MinMaxScaler()
  tscaler.min_, tscaler.scale_ = scaler.min_[1], scaler.scale_[1]
  preds = tscaler.inverse_transform(preds)
  test = tscaler.inverse_transform(test)
  plot.plot(np.arange(preds.shape[0]), preds[:, 0])
  plot.plot(np.arange(test.shape[0]-window_size), test[window_size:, 0])
  plot.show()

###################################################################################
# Format

print('Loading data...')
df = load_data()

print('Total Data Points: {}'.format(len(df.time)))

###################################################################################
# Visualize

# graph_whole(df)
# graph_years(df)
# graph_months(df)
# graph_days(df)
# points_per_month(df)

###################################################################################
# Format and Standardize

FEATURES = 1
WINDOW_SIZE = 12
TRAIN_TEST_SPLIT = 0.8
NODES = 8
EPOCHS = 100
BATCH_SIZE = 500
OPTIMIZER = optimizers.Adam()
LOSS = losses.mean_squared_error

years = split_years(df)
months = split_months(years[9])
x_train, y_train, x_test, y_test, tscaler = format_traintest_data(years[9], WINDOW_SIZE, TRAIN_TEST_SPLIT)

# https://machinelearningmastery.com/multi-step-time-series-forecasting-long-short-term-memory-networks-python
# https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python

###################################################################################
# Model

print('Running network...')
model = build_network(NODES, WINDOW_SIZE, FEATURES, OPTIMIZER, LOSS)
train_network(model, x_train, y_train, EPOCHS, BATCH_SIZE)

#for i in range(len(years)):

x_valid, y_valid, vscaler = format_validation_data(years[21], WINDOW_SIZE)
predict_into_future(model, y_valid, WINDOW_SIZE, vscaler)
#test_network(model, x_valid, y_valid, vscaler)



# # Un-Normalize
# x_test = x_test[:, 0]
# y_test = y_test[:, 0]
# test = np.array(pd.DataFrame({ 'time': y_test, 'temp': x_test }))
# test = scaler.inverse_transform(test)

# df = pd.DataFrame({ 'time': test[:, 0], 'temp': test[:, 1] })
# df = df.set_index('time')
# axis = df.plot.line(grid=True)
# axis.set_yticklabels(['{}C'.format(y) for y in axis.get_yticks()])
# axis.set_ylabel('Temperature in C')
# axis.set_xticklabels([datetime.utcfromtimestamp(x).strftime('%Y-%m-%d') for x in axis.get_xticks()])
# axis.set_xlabel('Time (1996 - 2019)')
# axis.get_legend().remove()
# plot.show()

# x_epochs = [x for x in range(EPOCHS)]
# plot.plot(x_epochs, history.history['loss'])
# plot.plot(x_epochs, history.history['mean_absolute_error'])
# plot.show()