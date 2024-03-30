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


### TUNING & CONFIG

# Lookback 
# This option will cause computing lags for each variable in the data.
# These lags will be fed into the LSTM model, toghether with the 
# original variables. 
hourly_lookback = 72           # Hourly data for three days
daily_lookback = 7             # Daily aggregated data for 6 days back
weekly_lookback = 52           # Weekly aggregated data for a year


### CONSTANTS 

input_data_loc = os.path.join("original_data", "airpollution", "LSTM-Multivariate_pollution.csv")
# The random component is so we don't overwrite our transformations each 
# time we tune the process
output_data_loc = os.path.join("transformed_data", "lstm_input-%d.csv"%rnd.randint(0, 1000))

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

## Daily and monthly

data['month']= pd.to_datetime(data['date']).dt.month
data['year']=pd.to_datetime(data['date']).dt.year
data['day']=pd.to_datetime(data['date']).dt.day
data['week']=pd.to_datetime(data['date']).dt.isocalendar().week                


daily_average_pollution = data.groupby(['year', 'month', 'day'])['pollution'].transform('mean')
data['d_avg_pollution']=daily_average_pollution

weekly_average_pollution = data.groupby(['year', 'week'])['pollution'].transform("mean")
data['w_avg_pollution'] = weekly_average_pollution

monthly_average_pollution = data.groupby(['year', 'month'])['pollution'].transform('mean')
data['m_avg_pollution']=monthly_average_pollution


data.to_csv(output_data_loc)

        



