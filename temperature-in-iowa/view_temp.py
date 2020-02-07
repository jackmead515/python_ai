import matplotlib.pyplot as plot
import pandas as pd
import numpy as np
import math
import time
from datetime import datetime

import util

def graph_years(df):
  years = np.array(util.split_years(df))
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

def graph_year(df):
  years = np.array(util.split_years(df))
  year = np.array(years[5])
  figure = plot.figure(facecolor='white')
  plot.plot(year[:, 0], year[:, 1], color=[0.0, 1.0, 0.0])
  plot.plot(year[:, 0], year[:, 2], color=[0.0, 0.0, 1.0])
  plot.show()

def graph_months(df):
  years = util.split_years(df)
  months = np.array(util.split_months(years[9]))
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
  years = util.split_years(df)
  months = util.split_months(years[9])
  days = np.array(util.split_days(months[5]))
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
  years = util.split_years(df)
  months = np.array(util.split_months(years[9]))
  lengths = []
  for i in range(len(months)):
    lengths.append(len(months[i]))
    print('Month: ' + str(i) + ' data points: ' + str(len(months[i])))
  print('Median: ' + str(np.median(lengths)))
  print('Mode: ' + str(np.average(lengths)))

if __name__ == "__main__":
  print('Loading data...')
  df = util.load_temp_data()

  print('Total Data Points: {}'.format(len(df.time)))
  # graph_whole(df)
  # graph_years(df)
  # graph_year(df)
  # graph_months(df)
  # graph_days(df)
  # points_per_month(df)
