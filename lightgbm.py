import numpy as np
from sklearn.preprocessing import MinMaxScaler
import cmath
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns


### TUNING AND CONFIG

# The location of the input file
transformed_data_loc = os.path.join("transformed_data", "lstm_input-final.csv")
original_data_log = os.path.join("original_data", "airpollution", "LSTM-Multivariate_pollution.csv")

# Time horizon of the prediction
horizon  = 24

# Lookback on which we make prediction 
lookback = 24

# Validation steps
stops = np.arange((-10)*horizon, 0, horizon) 



### IMPORT DATA
data = pd.read_csv(transformed_data_loc, sep=',', decimal='.')



### REMOVE UNNECESSARY COLUMNS

# We don't need the date
data = data.drop(columns=['date', 'month', 'year', 'day', 'week'])


# We don't need indices too
data = data.drop(data.columns[0], axis=1)

### CONVERT TO NUMPY ARRAYS AND RESHAPE
# Note: this is probably the most difficult part of the scripts altogether.
# We need to transform 2D array of num_observations x num_features into 
# 3D array of num_windows x num_observations_per_window x num_features.
# Using sliding window, which is a number of past observations we feed into 
# the LSTM at each step. And of course, it's the length back we need to 
# look, so we can make prediction with the already trained network

# Here we create windows from X, but we like the last window to be 
# the one that ends with the values, which are exactly horizon rows 
# from the end, because we will forecast this horizon.
# Hence [:-horizon] part.

X = data.drop(columns=['pollution'])[:-horizon].values
print(X.shape)
tmp = np.lib.stride_tricks.sliding_window_view(X, axis=0, window_shape=lookback)
print(tmp.shape)
X = np.swapaxes(tmp, 1, 2)
print(X.shape)

# Here we create windows from Y, but we want them to start after 
# our first lookback window. Hence the [lookback:] part.
Y = data['pollution'][lookback:].values
tmp = np.lib.stride_tricks.sliding_window_view(Y, axis=0, window_shape=horizon)
Y = tmp
print(Y.shape)


# We now separate training from the testing set
# The idea is to "shorten" our sets so the last Y window 
# of horizon observations ends at the last element of the 
# testing set. 
trainX = X[:-horizon]
print(trainX.shape)
trainY = Y[:-horizon]


testX = X[-horizon:]
print(testX.shape)
testY = Y[-horizon:]

### CONSTRUCT MODEL
model = Sequential()
model.add(LSTM(layer1Neurons, input_shape=(None,10), activation='tanh', return_sequences=False))
#model.add(LSTM(layer2Neurons, activation='tanh', return_sequences=True))
#model.add(LSTM(layer2Neurons, activation='tanh', return_sequences=True))
#model.add(LSTM(layer2Neurons, activation='tanh', return_sequences=False))

model.add(Dense(units=horizon, activation='tanh'))
model.compile(loss='mean_squared_error', optimizer='sgd')

#Shuffle shouldn't matter, we are feeding the net whole windows
model.fit(trainX, trainY, epochs=epochs, validation_data=(testX, testY), shuffle=True)


### Making prediction 
predictX = X[-1:]
print(predictX.shape)
predictY = model.predict(predictX)
print(predictY.shape)

fig = plt.figure(figsize=(16,9))
plt.plot(testY[-1:][0], color='blue')
plt.plot(predictY[0], color='green')
fig.show()    
