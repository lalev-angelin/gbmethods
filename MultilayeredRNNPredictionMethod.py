#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 10:17:02 2023

@author: ownjo
"""

import numpy as np
from PredictionMethod import PredictionMethod
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM
from sklearn.preprocessing import MinMaxScaler
import sys
import cmath

class MultilayeredRNNPredictionMethod(PredictionMethod):
    
    def __init__(self, data, numTrainPoints, numSeasons=1, numNeurons=(None, None, None)):
        super().__init__(data, numTrainPoints, numSeasons)
        self.numNeurons = numNeurons
    
    def constructModel(self):
        if self.numNeurons[0] == None: 
           rnn1NeuronCount = self.numAllPoints * 2
           model = Sequential()
           model.add(SimpleRNN(20, input_shape=(1,1), return_sequences=True, activation='tanh'))
           model.add(SimpleRNN(20, return_sequences=True, activation='tanh'))
           model.add(SimpleRNN(1, return_sequences=True, activation='tanh'))

           model.compile(loss='mean_squared_error', optimizer='adam')
        else:
           sys.exit(1)
           
        return model
    
    def predict(self):
        npData = np.array(self.data)
        #print(npData)
       
        scaler = MinMaxScaler(feature_range=(-1,1))

        # Мащабиране на обучителното множество
        npScaledData = npData.reshape(-1, 1)
        npScaledData = scaler.fit_transform(npScaledData)
        npScaledData = npScaledData.reshape(-1)
        
        # Не можем да влизаме в тестовото множество
        # a искаме да правим прогноза, обхващаща 
        # numTestPoints периода напред
        npTrainInput = np.array(npScaledData[:-2*self.numTestPoints]).reshape(-1,1)

        # Оформя изхода на групи от по predictions_plus_gap показатели
        npBatchedOutput = np.array([npScaledData[i:i+self.numTestPoints] for i in range(0, self.numAllPoints-self.numTestPoints+1)])
        
        npTrainOutput = np.array(npBatchedOutput[1:-self.numTestPoints])
        
        
        npTestInput = np.array(npScaledData[:-self.numTestPoints]).reshape(-1,1)
       
        
        model = self.constructModel()
        model.fit(x=npTrainInput, y=npTrainOutput, epochs=2000)
        
        rawPredicted = model.predict(npTestInput)
       
        predictedFirst = np.take(rawPredicted, 0, axis=1)

        #print("Predicted_first dimensions:", predicted_first.shape)
        #print(predicted_first)

        predictedLast = np.array(rawPredicted[-1:,1:].reshape(-1))
        #print("Predicted_last dimensions:", predicted_last.shape)
        #print("Predicted_last:", predicted_last)

        predicted = np.concatenate((predictedFirst, predictedLast))

        predicted = predicted.reshape(-1,1)
        predicted = scaler.inverse_transform(predicted)
        predicted = predicted.reshape(-1)
        predicted = np.insert(predicted, 0, np.NaN)
        
        self.prediction = predicted.flatten()
        
        return self.prediction

    
