#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 22:15:25 2023

@author: ownjo
"""
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from MovingAveragePredictionMethod import MovingAveragePredictionMethod
from ExponentialPredictionMethod import ExponentialPredictionMethod
from DoubleExponentialPredictionMethod import DoubleExponentialPredictionMethod
from TripleExponentialPredictionMethod import TripleExponentialPredictionMethod
from SimpleRNNEnsemblePredictionMethod import SimpleRNNEnsemblePredictionMethod
from SimpleRNNPredictionMethod import SimpleRNNPredictionMethod
from SimpleLSTMPredictionMethod import SimpleLSTMPredictionMethod
from SimpleLSTMEnsemblePredictionMethod import SimpleLSTMEnsemblePredictionMethod
from CombinedRNNPredictionMethod import CombinedRNNPredictionMethod
from AveragedRNNExponentialPredictionMethod import AveragedRNNExponentialPredictionMethod
from FeedForwardPredictionMethod import FeedForwardPredictionMethod
from DetrendingDecorator import DetrendingDecorator
from BoxCoxDecorator import BoxCoxDecorator

import pandas as pd
import os
import sys

### ЗАРЕЖДАНЕ НА ДАННИТЕ

data = pd.read_csv('gbmethods\original_data\airpollution\LSTM-Multivariate_pollution.csv', sep=';', decimal=".")
print(data.head)
sys.exit(1)

# Ако искаме всички графики в един pdf
# pdf = PdfPages("results/graphs.pdf")

for q in range (0, len(data)):

    # Номер на ред с данни
    rowno = q
    
    # Вземаме ред с данни
    row = data.iloc[rowno,7:].dropna()
 
    # Този фикс не беше нужен, когато работихме под Linux
    # Различна версия на библиотеките? Различно CSV?
    row = row.astype('float')
 
    # datapoints ще съдържа броя на наблюденията
    datapoints = data.iloc[rowno,1]
    print("datapoints:", datapoints)
    
    # Брой сезони, 1 - ако няма сезонност
    seasonality = data.iloc[rowno, 2]

    # future_predictions ще съдържа броя на периодите в бъдещето, които 
    # трябва да прогнозираме
    future_predictions = data.iloc[rowno,3]
    print("future_predictions:", future_predictions)
    
    # series_name ще съдържа името на серията
    series_name = data.iloc[rowno,0].lstrip().rstrip()
    print("series_name:", series_name)
    
    # series_type ще съдържа типа на серията - годишна, месечна ... 
    series_type=data.iloc[rowno,4]
    print("series_type", series_type)
    
    # Някои проверки за валидност 
    assert datapoints + future_predictions == len(row.values)
    

    # Проверяваме дали вече не сме го правили 
    if os.path.isdir("results/"+series_name):
        if (os.path.exists("results/"+series_name+"/fig2.png")):
            continue
        #continue
    else:
        os.mkdir("results/"+series_name)

   
    # Движеща се средна
    average = MovingAveragePredictionMethod(row, datapoints, window=3)
    prediction = average.predict()
        
    
    fig = plt.figure(q, figsize=(28,15))
    plt.grid(True, dashes=(1,1))
    plt.title(series_name+series_type)
    plt.xticks(rotation=90)
    plt.plot(average.data, color="blue", label="Original data")
    plt.axvline(x=datapoints, color="red", linestyle="--", label="Forecast horizon")
    plt.plot(prediction, color="orange", label="Moving average 3, MAPE=%s, wMAPE=%s" % (average.computeMAPE(), average.computeWMAPE()))
    plt.legend()
    
    average.save("results/"+series_name+"/mav.json")

    # # Просто експоненциално изглаждане
    # exponential = ExponentialPredictionMethod(row, datapoints)
    # prediction = exponential.predict()
    # plt.plot(prediction, color="magenta", label="Exp. smoothing $\\alpha=%s$, MAPE=%s, wMAPE=%s" % 
    #           (exponential.hwresults.params['smoothing_level'], exponential.computeMAPE(), exponential.computeWMAPE()))
    # plt.legend()

    # # Двойно експоненциално изглаждане
    # holt = DoubleExponentialPredictionMethod(row, datapoints)
    # prediction = holt.predict()
    # plt.plot(prediction, color="black", label="Exp. smoothing $\\alpha=%s$, $\\beta=%s$, MAPE=%s, wMAPE=%s" % 
    #           (holt.hwresults.params['smoothing_level'], holt.hwresults.params['smoothing_trend'], holt.computeMAPE(), holt.computeWMAPE()))
    # plt.legend()

    # Тройно експоненциално изглаждане. Ако имаме сезонни данни, тройно. Ако не - двойно.
    if seasonality>1:
        holtwinters = TripleExponentialPredictionMethod(row, datapoints, numSeasons=seasonality)
    else:
        holtwinters = DoubleExponentialPredictionMethod(row, datapoints)
    prediction = holtwinters.predict()
    # holtwinters.save("results/"+series_name+"/hw.json")
    plt.plot(prediction, color="red", label="Exp. smoothing $\\alpha=%s$, $\\beta=%s$ $\\gamma=%s$, MAPE=%s, wMAPE=%s" % 
              (holtwinters.hwresults.params['smoothing_level'], holtwinters.hwresults.params['smoothing_trend'], 
                holtwinters.hwresults.params['smoothing_seasonal'], holtwinters.computeMAPE(), holtwinters.computeWMAPE()))
    plt.legend()

    # Проста RNN Ensemble
    # simpleRNNEnsemble = SimpleRNNEnsemblePredictionMethod(row, datapoints, window=3)
    # prediction = simpleRNNEnsemble.predict()
    # simpleRNNEnsemble.save("results/"+series_name+"/srnn_ensemble.json")
    # simpleRNNEnsemble.saveModel("results/"+series_name+"/srnn_ensemble", "keras")
    # plt.plot(prediction, color="green", label="Simple RNN Ensemble, MAPE=%s, wMAPE=%s" % (simpleRNNEnsemble.computeMAPE(), simpleRNNEnsemble.computeWMAPE()))
    # plt.legend()
    
    
    # # Проста RNN 
    # simpleRNN = SimpleRNNPredictionMethod(row, datapoints, window=3)
    # prediction = simpleRNN.predict()
    # simpleRNN.save("results/"+series_name+"/srnn.json")
    # simpleRNN.saveModel("results/"+series_name+"/srnn.keras")
    # plt.plot(prediction, color="brown", label="Simple RNN, MAPE=%s, wMAPE=%s" % (simpleRNN.computeMAPE(), simpleRNN.computeWMAPE()))
    # plt.legend()
    
    # Проста LSTM
    # simpleLSTM = SimpleLSTMPredictionMethod(row, datapoints, window=3)
    # prediction = simpleLSTM.predict()
    # simpleLSTM.save("results/"+series_name+"/slstm.json")
    # simpleLSTM.saveModel("results/"+series_name+"/slstm.keras")
    # plt.plot(prediction, color="brown", label="Simple LSTM, MAPE=%s, wMAPE=%s" % (simpleLSTM.computeMAPE(), simpleLSTM.computeWMAPE()))
    # plt.legend()
    
    # Проста LSTM Ensemble
    simpleLSTMEnsemble = SimpleLSTMEnsemblePredictionMethod(row, datapoints, window=3)
    prediction = simpleLSTMEnsemble.predict()
    simpleLSTMEnsemble.save("results/"+series_name+"/slstm_ensemble_w3.json")
    simpleLSTMEnsemble.saveModel("results/"+series_name+"/slstm_ensemble_w3", "keras")
    plt.plot(prediction, color="brown", label="Simple LSTM Ensemble (window=3), MAPE=%s, wMAPE=%s" % (simpleLSTMEnsemble.computeMAPE(), simpleLSTMEnsemble.computeWMAPE()))
    plt.legend()
    
    simpleLSTMEnsemble = SimpleLSTMEnsemblePredictionMethod(row, datapoints, window=6)
    prediction = simpleLSTMEnsemble.predict()
    simpleLSTMEnsemble.save("results/"+series_name+"/slstm_ensemble_w6.json")
    simpleLSTMEnsemble.saveModel("results/"+series_name+"/slstm_ensemble_w6", "keras")
    plt.plot(prediction, color="green", label="Simple LSTM Ensemble (window=6), MAPE=%s, wMAPE=%s" % (simpleLSTMEnsemble.computeMAPE(), simpleLSTMEnsemble.computeWMAPE()))
    plt.legend()

    # Комбиниран RNN
    # combinedRNN = CombinedRNNPredictionMethod(row, datapoints, numSeasons=seasonality)
    # prediction = combinedRNN.predict()
    # combinedRNN.save("results/"+series_name+"/crnn.json")
    # combinedRNN.saveModel("results/"+series_name+"/crnn.keras")
    # plt.plot(prediction, color="darkgray", label="Combined RNN, MAPE=%s, wMAPE=%s" % (combinedRNN.computeMAPE(), combinedRNN.computeWMAPE()))
    # plt.legend()
    
    # Усреднен RNN
    # averagedRNN = AveragedRNNExponentialPredictionMethod(row, datapoints, numSeasons=seasonality)
    # prediction = averagedRNN.predict()
    # averagedRNN.save("results/"+series_name+"/arnn.json")
    # averagedRNN.saveModel("results/"+series_name+"/crnn.keras")
    # plt.plot(prediction, color="yellow", label="Averaged RNN, MAPE=%s, wMAPE=%s" % (averagedRNN.computeMAPE(), averagedRNN.computeWMAPE()))
    # plt.legend()
    
    # RNN с детрендинг
    # detrendedRNN = DetrendingDecorator(SimpleRNNPredictionMethod(row, datapoints))
    # prediction = detrendedRNN.predict()
    # detrendedRNN.save("results/"+series_name+"/detrended_rnn.json")
    # detrendedRNN.saveModel("results/"+series_name+"/crnn.keras")
    # plt.plot(prediction, color="green", linestyle="--", label="Detrended RNN, MAPE=%s, wMAPE=%s" % (detrendedRNN.computeMAPE(), detrendedRNN.computeWMAPE()))
    # plt.legend()
    
    # RNN с Бокс-Кокс
    # boxcoxRNN = BoxCoxDecorator(SimpleRNNPredictionMethod(row, datapoints))
    # prediction = boxcoxRNN.predict()
    # boxcoxRNN.save("results/"+series_name+"/boxcox_rnn.json")
    # plt.plot(prediction, color="green", linestyle="-.", label="BoxCox RNN, MAPE=%s, wMAPE=%s" % (boxcoxRNN.computeMAPE(), boxcoxRNN.computeWMAPE()))
    # plt.legend()
    
    plt.savefig("results/"+series_name+"/fig1.png")
    # pdf.savefig(fig)


#pdf.close()
