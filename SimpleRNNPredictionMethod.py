#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 10:17:02 2023

@author: ownjo
"""

import numpy as np
from NeuralNetworkPredictionMethod import NeuralNetworkPredictionMethod
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM
from sklearn.preprocessing import MinMaxScaler
import sys
import cmath

class SimpleRNNPredictionMethod(NeuralNetworkPredictionMethod):
    
    def __init__(self, data, numTrainPoints, numSeasons=1, numNeurons=(None, None, None), window=2):
        super().__init__(data, numTrainPoints, numSeasons)
        self.numNeurons = numNeurons
        self.window = window
       
    def getParameters(self): 
        params = {}
        params['extended_name']="Simple Recurrent Neural Network"
        params['layers']=self.layer_description
        params['window']=self.window
        return params       
     
    def constructModel(self):
        if self.numNeurons[0] == None: 
           rnn1NeuronCount = round(self.numAllPoints / 2)
           model = Sequential()
           model.add(SimpleRNN(rnn1NeuronCount, input_shape=(self.window,1), activation='tanh'))
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
        
        batchedInData = np.array([])
        for j in range(0, len(npScaledData)-self.window+1):
            tmp = []
            for w in range(j, j+self.window): 
                tmp.append(npScaledData[w])
            batchedInData=np.append(batchedInData, tmp)
        batchedInData = batchedInData.reshape(-1,self.window)     
        #print("Batched in data:\n", batchedInData)
        
        batchedTrainInData = batchedInData[ : -2*self.numTestPoints]
        batchedTestInData = batchedInData[-self.numTestPoints-1 : -self.numTestPoints]
        #print("Batched train data:\n", batchedTrainInData)
        #print("Batched test data:\n", batchedTestInData)

        batchedOutData = np.array([])
        for j in range(self.window, len(npScaledData)-self.numTestPoints+1):
            tmp = []
            for w in range(j, j+self.numTestPoints): 
                tmp.append(npScaledData[w])
            batchedOutData=np.append(batchedOutData, tmp)
        batchedOutData = batchedOutData.reshape(-1,self.numTestPoints)     
        #print("Batched out data dimensions:\n", batchedOutData.shape)
        #print("Batched out data:\n", batchedOutData)
        batchedOutTrainData = batchedOutData [:-self.numTestPoints]
        batchedOutTestData = batchedOutData[:-1]
        
        self.model = self.constructModel()
        self.model.fit(x=batchedTrainInData, y=batchedOutTrainData, epochs=1000, use_multiprocessing=True)
       
        predictionBase = batchedInData[:-self.numTestPoints]
        #print("predictionBase dimensions\n", predictionBase.shape)
        #print("predictionBase\n", predictionBase)
        
        rawPredicted = self.model.predict(predictionBase)
        #print("rawPredicted dimensions\n", rawPredicted.shape)
        #print("rawPredicted\n", rawPredicted)
        predictedFirst = np.take(rawPredicted, 0, axis=1)
 

        #print("Predicted_first dimensions:", predictedFirst.shape)
        #print(predictedFirst)

        predictedLast = np.array(rawPredicted[-1:,1:]).reshape(-1)
        #print("Predicted_last dimensions:", predictedLast.shape)
        #print("Predicted_last:", predictedLast)

        predicted = np.concatenate((predictedFirst, predictedLast))
        #print("Predicted dimensions:", predicted.shape)
        #print("Predicted:", predicted)

        predicted = predicted.reshape(-1,1)
        predicted = scaler.inverse_transform(predicted)
        predicted = predicted.reshape(-1)
        for i in range(0, self.window): 
            predicted = np.insert(predicted, 0, np.NaN)
        
        self.prediction = predicted.flatten()
        
        return self.prediction

    
