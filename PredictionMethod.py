#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 20:22:45 2023

@author: ownjo
"""

import json
from OurJSONEncoder import OurJSONEncoder

class PredictionMethod: 

    ##############################################################
    # Създава обучително и тестово множество 
    # 
    
    def formatData(self):
        pass    


    ##############################################################
    # Изчислява прогноза за тестовото множество
    # 
    def predict(self):
        pass
    
    def toJSON(self):
        saveData = {}
        saveData['method']=self.__class__.__name__
        saveData['parameters']=self.getParameters()
        saveData['numDataPoints']=self.numAllPoints
        saveData['numTrainPoints']=self.numTrainPoints
        saveData['numTestPoints']=self.numTestPoints
        saveData['seasons']=self.numSeasons
        saveData['data']=list(self.data)
        saveData['prediction']=list(self.prediction)
        saveData['MAPE']=self.computeMAPE()
        saveData['wMAPE']=self.computeWMAPE()
        return json.dumps(saveData, indent=2, cls=OurJSONEncoder)
    
    ##############################################################
    # Записва данни за модела в JSON формат
    #
    def save(self, filename):
        str = self.toJSON() 
        file = open(filename, "w")
        file.write(str)
        file.close
        
    ##############################################################
    # Зарежда модела
    #
    def load(self): 
        pass
      
    ##############################################################
    # Изчислява средна MAPE за прогнозата.
    #
    def computeMAPE(self): 
        lst1 = self.testData
        lst2 = self.prediction[-self.numTestPoints:]
        assert len(lst1)==len(lst2)

        mape = 0
        for i in range(0, len(lst1)):
           mape = mape + abs((lst1[i]-lst2[i])/lst1[i])
 
        
        return mape/len(lst1)
    
    def computeWMAPE (self):
        lst1 = self.testData
        lst2 = self.prediction[-self.numTestPoints:]
        assert len(lst1)==len(lst2)

        sumdiff = 0
        div = 0
        for i in range(0, len(lst1)):
            sumdiff = sumdiff + abs(lst1[i]-lst2[i])
            div = div + abs(lst1[i])
        
        return sumdiff/div
        
    
    ##############################################################
    # Връща текст с описание на прогностичния метод и параметри
    # 
    def describe(self):
        return ""
             
    ##########################################################
    # Конструктор
    # data - списък с данни
    # numTrainPoints - брой елементи на обучителното множество
    # numSeasons - брой сезони за сезонните данни
    
    def __init__(self, data, numTrainPoints, numSeasons=1):
        self.data = data
         
        self.numAllPoints=len(data)
        assert(self.numAllPoints>0)
         
        self.numTrainPoints=numTrainPoints
        assert(self.numTrainPoints>0)
        assert(self.numTrainPoints<self.numAllPoints)
         
        self.numTestPoints = self.numAllPoints - self.numTrainPoints
        self.numSeasons=numSeasons

        self.trainData = data[0:self.numTrainPoints]
        self.testData = data[self.numTrainPoints:]

    