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
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM

### TUNING & CONFIG

# Print all the graphs to PDF file (results/graphs.pdf)
print_to_pdf = False
# Print all the graphs to files  
print_to_file = False

### CONSTANTS 

input_data_loc = os.path.join("original_data", "airpollution", "LSTM-Multivariate_pollution.csv")
pdf_file_loc = os.path.join("results", "graphs.pdf")
image_file_loc = os.path.join("results");

### INITIALIZATION

if print_to_pdf:
    pdf = PdfPages(pdf_file_loc)


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

#### EXPLORATORY ANALYSIS AND VISUALIZATIONS

## Autocorrelation matrix
# print(dataNoTime.corr())

## ACF Plots
#for lag in [50, 1000, 20000]:
#    fig = plt.figure(figsize=(800, 600))
#    fig.title="ACF lags"
#    tsa.plot_acf(data['pollution'], lags=lag)
#    fig.show()
#    if print_to_pdf:
#        pdf.savefig(fig) 
#    if print_to_file:
#        fig.savefig(os.path.join(image_file_loc, "fig%d.png" % lag))


## PACF Plots
#for lag in [50, 1000, 20000]:
#    fig = plt.figure()
#    fig.title="PACF lags"
#    tsa.plot_pacf(data['pollution'], lags=lag)
#    fig.show()

#fig=plt.figure()
#tsa.plot_pacf(data['pollution'], lags=20000)
#fig.show()


if print_to_pdf: 
    pdf.close()


npData = np.array(dataNoTime)
print(npData)
sys.exit(0)




    
    
#plt.savefig("results/"+series_name+"/fig1.png")
# pdf.savefig(fig)
#pdf.close()
