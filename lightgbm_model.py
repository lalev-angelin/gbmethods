import numpy as np
from sklearn.preprocessing import MinMaxScaler
import cmath
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMRegressor

### TUNING AND CONFIG

# The location of the input file
transformed_data_loc = os.path.join("transformed_data", "lstm_input-final.csv")
original_data_log = os.path.join("original_data", "airpollution", "pollution.csv")

skip_first_n_rows = 48

### RELEVANT COLUMNS
### |Corrleation coeff| >0.2 

columns = ['wnd_spd', "wnd_spd-1", "wnd_spd-2", "wnd_spd-3", "wnd_spd-4", "wnd_spd-5", 
           "wnd_spd-6", "wnd_spd-7", "wnd_spd-8", "wnd_spd-9",
           "pollution-1", "pollution-2", "pollution-3", "pollution-4", "pollution-5", 
           "pollution-6", "pollution-7", "pollution-8", "pollution-9", "pollution-10", 
           "pollution-11", "pollution-12", "pollution-13", "pollution-14", "pollution-15", 
           "pollution-16", "pollution-17", "pollution-18", "pollution-19", "pollution-20", 
           "pollution-21", "pollution-22",  "pollution-23", "d_avg_pollution-1"]


### IMPORT DATA
data = pd.read_csv(transformed_data_loc, sep=',', decimal='.')

### SPLIT 

X = data[columns][skip_first_n_rows:]
Y = data['pollution'][skip_first_n_rows:]

trainX=X[:-24]
trainY=Y[:-24]
print(trainX)
print(trainY)

testX=X[-24:]
testY=Y[-24:]
print(testX)
print(testY)

### FIT

regressor = LGBMRegressor(boosting_type="gbdt", num_leaves=31,
                                   max_depth=-1, learning_rate=0.1, 
                                   n_estimators=1000)

regressor.fit(trainX, trainY)


### PREDICT
predictY = pd.DataFrame(regressor.predict(X))
predictY.rename({"0": "predicted_pollution"})
print(predictY)

### PLOT
fig = plt.figure() 
plt.plot(predictY-Y, color="green")
plt.show()

