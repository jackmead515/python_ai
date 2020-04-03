import matplotlib.pyplot as plot
import pandas as pd
import numpy as np
import math
import time
from datetime import datetime

def load_temp_data():
  print('Loading Temp Data...')
  df = pd.read_csv('ames_iowa_30-09-1996_26-10-2019_temp.csv')
  df = df.drop(['station'], axis=1)
  df = df.dropna()
  df.columns = ['time', 'temp']
  df.time = [datetime.strptime(x, "%Y-%m-%d %H:%M").timestamp() for x in df.time]
  print('Total Data Points: {}'.format(len(df.time)))
  return df

def load_humid_data():
  print('Loading Humid Data...')
  df = pd.read_csv('ames_iowa_30-09-1996_26-10-2019_temp_humid.csv')
  df = df.drop(['station'], axis=1)
  df = df.dropna()
  df.columns = ['time', 'temp', 'humid']
  df = df.drop(['temp'], axis=1)
  df.time = [datetime.strptime(x, "%Y-%m-%d %H:%M").timestamp() for x in df.time]
  print('Total Data Points: {}'.format(len(df.time)))
  return df

def load_temp_humid_data():
  print('Loading Temp and Humid Data...')
  df = pd.read_csv('ames_iowa_30-09-1996_26-10-2019_temp_humid.csv')
  df = df.drop(['station'], axis=1)
  df = df.dropna()
  df.columns = ['time', 'temp', 'humid']
  df.time = [datetime.strptime(x, "%Y-%m-%d %H:%M").timestamp() for x in df.time]
  print('Total Data Points: {}'.format(len(df.time)))
  return df

def split_years(df):
  values = np.array(df)
  years = [[]]
  index = 0
  current_year = datetime.fromtimestamp(values[0, 0]).year
  for pair in values:
    time = pair[0]
    data_year = datetime.fromtimestamp(time).year
    if data_year == current_year:
      years[index].append(np.array(pair))
    else:
      index += 1
      current_year = data_year
      years.append([])
  return years

def split_months(year):
  values = np.array(year)
  months = [[]]
  index = 0
  current_month = datetime.fromtimestamp(values[0, 0]).month
  for pair in values:
    time = pair[0]
    data_month = datetime.fromtimestamp(time).month
    if data_month == current_month:
      months[index].append(np.array(pair))
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
    data_day = datetime.fromtimestamp(time).day
    if data_day == current_day:
      days[index].append(np.array(pair))
    else:
      index += 1
      current_day = data_day
      days.append([])
  return days