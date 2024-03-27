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

class SimpleLSTMEnsemblePredictionMethod(NeuralNetworkPredictionMethod):
    
    def __init__(self, data, numTrainPoints, numSeasons=1, numNeurons=(None, None, None), window=2):
        super().__init__(data, numTrainPoints, numSeasons)
        self.numNeurons = numNeurons
        self.window = window
        self.numSeasons = numSeasons
        self.layer_description = []
       
    def getParameters(self): 
        params = {}
        params['extended_name']="Simple LSTM With \"Ensemble\" Predictions"
        params['window']=self.window
        params['numSeasons']=self.numSeasons
        return params       
     
    def constructModel(self, period):
       
        if self.numNeurons[0] == None: 
            for q in range(0, self.numTestPoints):
               lstm1NeuronCount = round(self.numAllPoints / 2)
               model = Sequential()
               model.add(LSTM(lstm1NeuronCount, input_shape=(self.window,1), activation='tanh'))
               model.add(Dense(units=1, activation='tanh'))
               model.compile(loss='mean_squared_error', optimizer='adam')
           
               self.layer_description={}
               self.layer_description['model' + str(period) + 'layer1_type']="SimpleRNN"
               self.layer_description['model' + str(period) + 'layer1_neuron_count']=lstm1NeuronCount
               self.layer_description['model' + str(period) + 'layer1_activation']="tanh"
               self.layer_description['model' + str(period) + 'layer2_type']="Dense"
               self.layer_description['model' + str(period) + 'layer2_neuron_count']=self.numTestPoints
               self.layer_description['model' + str(period) + 'layer2_activation']="tanh"
               self.layer_description['model' + str(period) + 'loss']="mean_squared_error"
               self.layer_description['model' + str(period) + 'optimizer']='adam'
               return model
           
        else:
           sys.exit(1)
           
        return None

    
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
        print("Batched in data:\n", batchedInData)

        self.models = []
        batchedTestInData = batchedInData[-self.numTestPoints - 1 : -self.numTestPoints]

        for q in range(0, self.numTestPoints):
        
            batchedTrainInData = batchedInData[ : -self.numTestPoints - q - 1]
            print("Batched train data:\n", batchedTrainInData)
            print("Batched test data:\n", batchedTestInData)
    
            
            trainOutData = npScaledData[self.window + q : -self.numTestPoints ]
            print("trainOutData:\n", trainOutData)

            model = self.constructModel(q)
            model.fit(x=batchedTrainInData, y=trainOutData, epochs=1000, use_multiprocessing=True)
            self.models.append(model)
        
        
        predictionBase = batchedInData[:-self.numTestPoints]
        print("predictionBase dimensions\n", predictionBase.shape)
        print("predictionBase\n", predictionBase)
        
        rawPredicted = self.models[0].predict(predictionBase)
        print("rawPredicted dimensions\n", rawPredicted.shape)
        print("rawPredicted\n", rawPredicted)
        predictedFirst = np.take(rawPredicted, 0, axis=1)
 
        print("Predicted_first dimensions:", predictedFirst.shape)
        print("Predicted first", predictedFirst)


        predictedLast = []   
        for q in range(1, self.numTestPoints):
             predictedPoint = self.models[q].predict(batchedTestInData)
             print("predictedPoint", predictedPoint[0][0])
             predictedLast.append(predictedPoint[0][0])

        predictedLast = np.array(predictedLast)
        print("Predicted_last dimensions:", predictedLast.shape)
        print("Predicted_last", predictedLast)
    
        
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

    def saveModel(self, filename, extension):
        for q in range(0, self.numTestPoints):
            self.models[q].save(filename+str(q)+"."+extension)
