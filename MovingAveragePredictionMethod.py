#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 20:42:54 2023

@author: ownjo
"""

from PredictionMethod import PredictionMethod
import cmath 

class MovingAveragePredictionMethod(PredictionMethod):
     
    ##########################################################
    # Конструктор
    # data - списък с данни
    # numTrainPoints - брой елементи на обучителното множество
    # numSeasons - брой сезони за сезонните данни
    # window - прозорец за движещата се средна 
    
    def __init__(self, data, numTrainPoints, numSeasons=1, window=3):
        super().__init__(data, numTrainPoints, numSeasons=numSeasons)
        assert window>0
        assert window<self.numAllPoints
        self.window = window

    def getParameters(self): 
        params = {}
        params['extended_name']="Moving average"
        params['window']=self.window
        return params           
    
    def predict(self): 
        self.prediction = []
        
        for i in range(0, self.window):
            self.prediction.append(cmath.nan)
        
        for i in range(self.window, self.numTrainPoints):
            self.prediction.append(sum(self.data[i-self.window:i])/self.window)
            
        for i in range(self.numTrainPoints, self.numAllPoints):
            self.prediction.append(sum(self.prediction[i-self.window:i])/self.window)
            
        return self.prediction
    
        
    def describe(self):
        return "Moving average, window = "+self.window
        

    