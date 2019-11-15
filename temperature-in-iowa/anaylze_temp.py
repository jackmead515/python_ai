import matplotlib.pyplot as plot
import pandas as pd
import numpy as np
import math
from keras.preprocessing import sequence
from keras import models
from keras import layers
from keras import losses
from keras import metrics
from keras import optimizers
from datetime import datetime

def forecast(data, window_size):
  datax, datay = [], []
  for i in range(len(data)-window_size-1):
    datax.append(data[i:(i+window_size), 0])
    datay.append(data[i + window_size, 0])
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

###################################################################################
# Format

df = pd.read_csv('ames_iowa_30-09-1996_26-10-2019_temp.csv')
df = df.drop(['station'], axis=1)
df = df.dropna()
df.columns = ['time', 'value']
# df.time = pd.DatetimeIndex(df.time).astype(np.int64)
df.time = [datetime.strptime(x, "%Y-%m-%d %H:%M").timestamp() for x in df.time]

###################################################################################
# Visualize

# graph_whole(df)
# graph_years(df)
# graph_months(df)

###################################################################################
# Format and Standardize

# x_train, x_test, y_train, y_test = train_test_split(time_data, value_data, test_size=0.2, random_state=42)
# Why bring in the extra library? No nonsense...

WINDOW_SIZE = 100
TRAIN_TEST_SPLIT = 0.5

dataset = np.array(df)

# Get minimums and maximums for later use
x_max, x_min = max(dataset[:, 0]), min(dataset[:, 0])
y_max, y_min = max(dataset[:, 1]), min(dataset[:, 1])

# Normalize values
dataset[:, 0] = dataset[:, 0] / np.sqrt(np.sum(dataset[:, 0]**2))
dataset[:, 1] = dataset[:, 1] / np.sqrt(np.sum(dataset[:, 1]**2))

# Forecast values to timeseries perdictions
datax, datay = forecast(dataset, WINDOW_SIZE)

# Split into test and training set
tts = int(len(datax)*TRAIN_TEST_SPLIT)
x_train, y_train = np.array(datax[:tts]), np.array(datay[:tts])
x_test, y_test = np.array(datax[tts+1:]), np.array(datay[tts+1:])

# Expands the dimensions to watch what keras wants
x_train, y_train = np.expand_dims(x_train, axis=2), np.expand_dims(y_train, axis=2)
x_test, y_test = np.expand_dims(x_test, axis=2), np.expand_dims(y_test, axis=2)

print('x_train shape', x_train.shape)
print('y_train shape', y_train.shape)
print('x_test shape', x_test.shape)
print('y_test shape', y_test.shape)

# https://machinelearningmastery.com/multi-step-time-series-forecasting-long-short-term-memory-networks-python
# https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python


# print('x_train shape', x_train.shape)
# x_train, y_train = create_forecasted_dataset(dataset, LOOK_BACK)

# scaler = MinMaxScaler(feature_range=(0, 1))
# dataset = scaler.fit_transform(dataset)

# train_data = dataset[:int(len(dataset)*0.8)]
# test_data = dataset[int(len(dataset)*0.8):]

# x_train, y_train = create_dataset(train_data, 5)
# x_test, y_test = create_dataset(test_data, 5)

# x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
# x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

###################################################################################
# Model

EPOCHS = 1
BATCH_SIZE = 4000

model = models.Sequential()
# model.add(layers.Embedding(len(x_train), 64, input_length=LOOK_BACK))
model.add(layers.LSTM(32, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=False))
# model.add(layers.Dropout(0.2))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

# plot.plot(range(EPOCHS), history.history['loss'])
# plot.show()

xp_test = model.predict(x_test)

print('xp_test', xp_test.shape)
print('y_test', y_test.shape)

# Un-Normalize
xp_test = xp_test[:, 0]
yp_test = y_test[:, 0]
# xp_test = xp_test*x_max - xp_test*x_min + x_max
yp_test = yp_test*y_max - yp_test*y_min + y_max

# plot baseline and predictions
# plot.plot(dataset)
plot.plot(range(len(xp_test)), xp_test)
plot.show()

# EPOCHS = 100
# LEARNING_RATE = 0.005
# BATCH_SIZE = 2048

# model = models.Sequential()
# model.add(layers.Dense(400, activation = 'relu', input_shape = (1,)))
# model.add(layers.Dense(1))
# model.compile(
#   optimizer = optimizers.RMSprop(lr = LEARNING_RATE),
#   loss = losses.mean_squared_error,
#   metrics = [ metrics.mae ]
# )
# history = model.fit(x_train, y_train, epochs = EPOCHS, batch_size = BATCH_SIZE)

# x_epochs = [x for x in range(EPOCHS)]
# plot.plot(x_epochs, history.history['loss'])
# plot.plot(x_epochs, history.history['mean_absolute_error'])
# plot.show()