#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 09:26:35 2023

@author: ownjo
"""

from PredictionMethod import PredictionMethod
from statsmodels.tsa.api import Holt
import numpy as np
import math

class DoubleExponentialPredictionMethod(PredictionMethod):
    
    def __init__(self, data, numTrainPoints, numSeasons=1, params=(None, None)):
        super().__init__(data, numTrainPoints, numSeasons)
        self.alpha = params[0]
        self.beta = params[1]
        # Или двата параметъра, или нищо. Който иска просто изглаждане, има си клас.
        if self.alpha==None:
            assert self.beta==None
        else:
            assert self.beta!=None
        
    def getParameters(self): 
        params = {}
        params['extended_name']="Double exponential smoothing"
        params['smooth_alpha']=self.hwresults.params['smoothing_level']
        params['smooth_beta']=self.hwresults.params['smoothing_trend']
        params['smooth_gamma']=math.nan
        return params                
            
    def predict(self):
        npdata = np.array(self.trainData)
        
        if (self.alpha==None):
            self.hwresults = Holt(npdata, initialization_method='estimated').fit(optimized=True)    
        else:
            self.hwresults = Holt(npdata, initialization_method='estimated').fit(smoothing_level=self.alpha, trend_level=self.beta, optimized=False)
            
        self.alpha = self.hwresults.params['smoothing_level']
        self.beta = self.hwresults.params['smoothing_trend']
        self.gamma = self.hwresults.params['smoothing_seasonal']    

        self.prediction = self.hwresults.predict(start=0, end=self.numAllPoints-1)
        return self.prediction