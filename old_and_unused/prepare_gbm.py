# -*- coding: utf-8 -*-
"""
Copyright 2024 Angelin Lalev

This program is free software: you can redistribute it and/or modify it under 
the terms of the GNU General Public License as published by the 
Free Software Foundation, either version 3 of the License, or 
(at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT 
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with 
this program. If not, see <https://www.gnu.org/licenses/>.
"""

"""
Data preparation for the LSTM model.
"""

import pandas as pd
import os
import sys
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import statsmodels.graphics.tsaplots as tsa
import random as rnd
from dateutil.relativedelta import relativedelta 

### TUNING & CONFIG

# Lookback 
# This option will cause computing lags for each variable in the data.
# These lags will be fed into the LSTM model, toghether with the 
# original variables. 
hourly_lookback = 24           # Hourly data for pollution from n past hours 
daily_lookback = 7             # Daily aggregated data from n past days 
weekly_lookback = 53           # Weekly aggregated data for n past weeks 
monthly_lookback = 12          # Monthly aggregated data for n past months 

### CONSTANTS 

input_data_loc = os.path.join("original_data", "airpollution", "pollution.csv")
# The random component is so we don't overwrite our transformations each 
# time we tune the process
output_data_loc = os.path.join("transformed_data", "gbm_input-%d.csv"%rnd.randint(0, 1000))

### INITIALIZATION


### LOADING OF THE DATA
data = pd.read_csv(input_data_loc, sep=',', decimal=".")
#print(data.head)


### SCALING & TRANSFORMING
scaler = MinMaxScaler(feature_range=(-1,1))

# pollution
tmp = data['pollution'].to_numpy()
data['pollution'] = scaler.fit_transform(tmp.reshape(-1, 1))

# dew 
tmp = data['dew'].to_numpy()
data['dew'] = scaler.fit_transform(tmp.reshape(-1, 1))

# temp 
tmp = data['temp'].to_numpy()
data['temp'] = scaler.fit_transform(tmp.reshape(-1, 1))

# press 
tmp = data['press'].to_numpy()
data['press'] = scaler.fit_transform(tmp.reshape(-1, 1))

# wnd_dir 
data['wnd_dir'] = data['wnd_dir'].replace(to_replace="SE", value=0.1)
data['wnd_dir'] = data['wnd_dir'].replace(to_replace="NW", value=0.3)
data['wnd_dir'] = data['wnd_dir'].replace(to_replace="NE", value=0.6)
data['wnd_dir'] = data['wnd_dir'].replace(to_replace="cv", value=0.9)

# wnd_spd 
tmp = data['wnd_spd'].to_numpy()
data['wnd_spd'] = scaler.fit_transform(tmp.reshape(-1, 1))

# snow 
tmp = data['snow'].to_numpy()
data['snow'] = scaler.fit_transform(tmp.reshape(-1, 1))

# rain 
tmp = data['rain'].to_numpy()
data['rain'] = scaler.fit_transform(tmp.reshape(-1, 1))


## Lagged data
for i in range(1, hourly_lookback):
    data['pollution-%d' % i] = data['pollution'].shift(i);
for i in range(1, hourly_lookback):
    data['dew-%d' % i] = data['dew'].shift(i);
for i in range(1, hourly_lookback):
    data['temp-%d' % i] = data['temp'].shift(i); 
for i in range(1, hourly_lookback):
    data['press-%d' % i] = data['press'].shift(i); 
for i in range(1, hourly_lookback):
    data['wnd_spd-%d' % i] = data['wnd_spd'].shift(i); 
for i in range(1, hourly_lookback):
    data['snow-%d' % i] = data['snow'].shift(i); 
for i in range(1, hourly_lookback):
    data['rain-%d' % i] = data['rain'].shift(i); 

## Daily and monthly

data.insert(1, 'year', pd.to_datetime(data['date']).dt.year)
data.insert(2, 'month', pd.to_datetime(data['date']).dt.month)
data.insert(3, 'day', pd.to_datetime(data['date']).dt.day)
data.insert(4, 'week', pd.to_datetime(data['date']).dt.isocalendar().week)                
data.insert(5, 'weekday', pd.to_datetime(data['date']).dt.weekday)

daily_average_pollution = data.groupby(['year', 'month', 'day'])['pollution'].transform('mean')
data['d_avg_pollution']=daily_average_pollution

weekly_average_pollution = data.groupby(['year', 'week'])['pollution'].transform("mean")
data['w_avg_pollution'] = weekly_average_pollution

for i in range(1, daily_lookback):
    data['d_avg_pollution-%d' % i] = data['d_avg_pollution'].shift(i*24)

for i in range(1, weekly_lookback):
    data['w_avg_pollution-%d' % i] = data['w_avg_pollution'].shift(i*24*7) 
    data['w_avg_pollution-%d' % i] = data['w_avg_pollution-%d' % i].fillna(data['w_avg_pollution'][0])   

monthly_average_pollution = data.groupby(['year', 'month'])['pollution'].transform('mean')
data['m_avg_pollution']=monthly_average_pollution

for i in range(0, monthly_lookback-1): 
    prev_month = pd.to_datetime(data['date']).apply(lambda x: x-relativedelta(months=(i+1)))
    data.insert(1+2*i, 'month-%d_month'%(i+1), prev_month.dt.month) 
    data.insert(1+2*i+1, 'month-%d_year'%(i+1), prev_month.dt.year) 
    
mapa = pd.DataFrame(data.groupby(['year', 'month'])['pollution'].mean().reset_index())
mapa.rename(columns={'pollution': "tmp"}, inplace=True)
for i in range(1, monthly_lookback):
    data = pd.merge(data, mapa, left_on=['month-%d_month'%i, 'month-%d_year'%i], right_on=['month', 'year'], how='left')
    data.rename(columns={'tmp':'m_avg_pollution-%d'%i}, inplace=True)
    data.drop(data.columns[-3:-1].values, axis=1, inplace=True)
    data.drop(['month-%d_month'%i], axis=1, inplace=True)
    data.drop('month-%d_year'%i, axis=1, inplace=True)



data.to_csv(output_data_loc)

        



