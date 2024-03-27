#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 10:17:02 2023

@author: ownjo
"""

import numpy as np
from PredictionMethod import PredictionMethod
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import sys

class FeedForwardPredictionMethod(PredictionMethod):
    
    def __init__(self, data, numTrainPoints, numSeasons=1):
        super().__init__(data, numTrainPoints, numSeasons)
    
    def constructModel(self):
       model = Sequential()
       model.add(Dense(20, input_shape=(1,1), activation='tanh'))
       model.add(Dropout(0.1))
       model.add(Dense(20, activation='tanh'))
       model.add(Dropout(0.1))
       model.add(Dense(self.numTestPoints, activation='tanh'))
       model.compile(loss='mean_squared_error', optimizer='adam')
       
       self.layer_description={}
       self.layer_description['layer1_type']="Dense"
       self.layer_description['layer1_neuron_count']=20
       self.layer_description['layer1_activation']="tanh"
       self.layer_description['layer2_type']="Dropout"
       self.layer_description['layer2_dropout']="0.1"
       self.layer_description['layer3_type']="Dense"
       self.layer_description['layer3_neuron_count']=20
       self.layer_description['layer3_activation']="tanh"
       self.layer_description['layer4_type']="Dropout"
       self.layer_description['layer4_dropout']="0.1"
       self.layer_description['layer5_type']="Dense"
       self.layer_description['layer5_neuron_count']=self.numTestPoints
       self.layer_description['loss']="mean_squared_error"
       self.layer_description['optimizer']='adam'
       
       return model

    def getParameters(self): 
        params = {}
        params['extended_name']="Multy-layered feed-forward neural network"
        params['layers']=self.layer_description    
        return params   
    
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
       
        
        self.model = self.constructModel()
        self.model.fit(x=npTrainInput, y=npTrainOutput, epochs=2000)
        
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
        
        self.prediction = predicted.flatten()
        
        return self.prediction


    def saveModel(self, fileName):
        self.model.save(fileName)
    
