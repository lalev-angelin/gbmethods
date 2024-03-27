#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 19:01:45 2023

@author: ownjo
"""

import numpy as np
from NeuralNetworkPredictionMethod import NeuralNetworkPredictionMethod
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
import sys
import cmath
from DoubleExponentialPredictionMethod import DoubleExponentialPredictionMethod 
from TripleExponentialPredictionMethod import TripleExponentialPredictionMethod


class AveragedRNNExponentialPredictionMethod(NeuralNetworkPredictionMethod):
    
    def predict(self):
               
        if self.numSeasons==1: 
            smooth = DoubleExponentialPredictionMethod(self.data, self.numTrainPoints)        
        else:
            smooth = TripleExponentialPredictionMethod(self.data, self.numTrainPoints, numSeasons=self.numSeasons)
        
        smoothData = smooth.predict()
        
        self.alpha = smooth.alpha
        self.beta = smooth.beta
        self.gamma = smooth.gamma
        
        
        npData = np.array(self.data)
        #print(npData)
       
        scaler = MinMaxScaler(feature_range=(-1,1))

        # Мащабиране на обучителното множество
        npScaledData = npData.reshape(-1, 1)
        npScaledData = scaler.fit_transform(npScaledData)
        npScaledData = npScaledData.reshape(-1)
        
        # Не можем да влизаме в тестовото множество,
        # a искаме да правим прогноза, обхващаща 
        # numTestPoints периода напред
        npTrainInput = np.array(npScaledData[:-2*self.numTestPoints]).reshape(-1,1)

        # Оформя изхода на групи от по predictions_plus_gap показатели
        npBatchedOutput = np.array([npScaledData[i:i+self.numTestPoints] for i in range(0, self.numAllPoints-self.numTestPoints+1)])
        
        npTrainOutput = np.array(npBatchedOutput[1:-self.numTestPoints])
        
        
        npTestInput = np.array(npScaledData[:-self.numTestPoints]).reshape(-1,1)
       
        
        self.model = self.constructModel(self.numAllPoints)
        self.model.fit(x=npTrainInput, y=npTrainOutput, epochs=1000)
        
        rawPredicted = self.model.predict(npTestInput)
       
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
        
        average = (smoothData + predicted.flatten())/2
        
        self.prediction = average
        
        return self.prediction

    def constructModel(self, rnn1NeuronCount):
        model = Sequential()
        model.add(SimpleRNN(rnn1NeuronCount, input_shape=(1,1), activation='tanh'))
        model.add(Dense(units=self.numTestPoints, activation='tanh'))
        model.compile(loss='mean_squared_error', optimizer='adam')

        self.layer_description={}
        self.layer_description['layer1_type']="SimpleRNN"
        self.layer_description['layer1_neuron_count']=rnn1NeuronCount
        self.layer_description['layer1_activation']="tanh"
        self.layer_description['layer2_type']="Dense"
        self.layer_description['layer2_neuron_count']=self.numTestPoints
        self.layer_description['layer2_activation']="tanh"
        self.layer_description['loss']="mean_squared_error"
        self.layer_description['optimizer']='adam'
        
        return model

    def getParameters(self): 
        params = {}
        params['extended_name']="Average of Double/Triple (for seasonal data) Exponential smoothing and Simple RNN"
        params['smooth_alpha']=self.alpha    
        params['smooth_beta']=self.beta
        params['smooth_gamma']=self.gamma
        params['rnn_layers']=self.layer_description
    
        return params    
    
    def saveModel(self, fileName):
        self.model.save(fileName)
