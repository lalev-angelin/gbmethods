#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 19:48:27 2023

@author: ownjo
"""
import os
from PredictionMethod import PredictionMethod

class NeuralNetworkPredictionMethod(PredictionMethod):
    
  
    def saveModel(self, fileName):
        self.model.save(fileName)
