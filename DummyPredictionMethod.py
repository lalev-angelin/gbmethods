#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 20:38:27 2023

@author: ownjo
"""

from PredictionMethod import PredictionMethod

# Трябва ни, за да тестваме декораторите
class DummyPredictionMethod(PredictionMethod):
    def predict(self):
        self.prediction = self.data
        return self.prediction
    
    def getParameters(self): 
        params = {}
        params['extended_name']="Dummy prediction method for testing. Returns the data as prediction"
        return params       
    