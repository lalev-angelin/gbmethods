#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 22:15:25 2023

@author: ownjo
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


### TUNING & CONFIG

# Print all the graphs in PDF file (results/graphs.pdf)
print_to_pdf = True



### CONSTANTS 

input_data_loc = os.path.join("original_data", "airpollution", "LSTM-Multivariate_pollution.csv")
pdf_file_loc = os.path.join("results", "graphs.pdf")


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

dataNoTime = pd.DataFrame(data[['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']])

#### EXPLORATORY ANALYSIS AND DRAWINGS

## Autocorrelation matrix
# print(dataNoTime.corr())

## ACF Plots
#for lag in range(10, 1000, 30):
#    fig = plt.figure()
#    fig.title="ACF lags"
#    tsa.plot_acf(data['pollution'], lags=lag)
#    fig.show()


## PACF Plots
#for lag in range(10, 10020, 1000):
#    fig = plt.figure()
#    fig.title="PACF lags"
#    tsa.plot_pacf(data['pollution'], lags=lag)
#    fig.show()

fig=plt.figure()
tsa.plot_pacf(data['pollution'], lags=20000)
fig.show()
sys.exit(0)








npData = np.array(dataNoTime)
print(npData)
sys.exit(0)




    
    
#plt.savefig("results/"+series_name+"/fig1.png")
# pdf.savefig(fig)
#pdf.close()
