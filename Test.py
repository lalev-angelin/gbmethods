#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 20:56:13 2023

@author: ownjo
"""

from PredictionMethod import PredictionMethod
from MovingAveragePredictionMethod import MovingAveragePredictionMethod
from DummyPredictionMethod import DummyPredictionMethod
from DetrendingDecorator import DetrendingDecorator
from BoxCoxDecorator import BoxCoxDecorator
import unittest
import cmath
import math
import numpy

class Test(unittest.TestCase): 
    def test_PredictionMethod(self): 
        data=[0,1,2,3,4,5,6,7,8,9]
        prediction = PredictionMethod(data, 5)
        self.assertEqual(prediction.data, data)
        self.assertEqual(prediction.trainData,  [0,1,2,3,4])
        self.assertEqual(prediction.testData, [5,6,7,8,9])
        self.assertEqual(prediction.numAllPoints, 10)
        self.assertEqual(prediction.numTrainPoints, 5)
        self.assertEqual(prediction.numTestPoints, 5)
        
    def test_MovingAveragePredictionMethod(self):
        data=[0,1,2,3,4,5,6,7,8,9]
        method = MovingAveragePredictionMethod(data, 9, window=3)
        method.predict()
        self.assertEqual(method.prediction, [cmath.nan, cmath.nan, cmath.nan, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0])

    def test_Mape(self):
        data=[0,1,2,3,4,5,6,7,8,9]
        method = MovingAveragePredictionMethod(data, 7)
        pred = method.predict()
        self.assertEqual(method.computeMAPE(), 0.5906819517930629)
        self.assertEqual(method.computeWMAPE(), 0.5925925925925926)
        
    def test_PredictionMethodSave(self):
        data=[0,1,2,3,4,5,6,7,8,9]
        method = MovingAveragePredictionMethod(data, 7)
        pred = method.predict()
        method.save('test.json')
        
    def test_DummyPredictionMethod(self):
        data=[0,1,2,3,4,5,6,7,8,9]
        method = DummyPredictionMethod(data, 3)
        pred = method.predict()
        self.assertEqual(pred, [0,1,2,3,4,5,6,7,8,9])
        
    def test_DetrendingDecorator1(self):
        data=[0,1,2,3,4,5,6,7,8,9]
        output=[cmath.nan,cmath.nan,cmath.nan,3,4,5,6,7,8,9]
        method = DetrendingDecorator(MovingAveragePredictionMethod(data, 7))
        pred = method.predict()
        for i in range(0, len(data)):
            numpy.testing.assert_almost_equal(pred[i], output[i])
            
    def test_BoxCoxDecorator(self):
        data=[1, math.e, math.e**2, math.e**3, math.e**4, math.e**5, math.e**6, math.e**7, math.e**8, math.e**9]
        output=[math.nan, math.nan, math.nan, 2.7182803022865554, 7.389051950363702, 20.08552564621351, 54.59811937915036, 20.085514369245708, 28.03159866405369, 31.325843428792496]
        method = BoxCoxDecorator(MovingAveragePredictionMethod(data, 7))
        pred = method.predict()

        for i in range(0, len(data)):
             numpy.testing.assert_almost_equal(pred[i], output[i])        
        
if __name__ == '__main__':
    unittest.main()
