#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 21:39:59 2023

@author: ownjo
"""
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
from SimpleRNNPredictionMethod import SimpleRNNPredictionMethod
from SimpleLSTMPredictionMethod import SimpleLSTMPredictionMethod
from CombinedRNNPredictionMethod import CombinedRNNPredictionMethod
from AveragedRNNExponentialPredictionMethod import AveragedRNNExponentialPredictionMethod
import pandas as pd

### ЗАРЕЖДАНЕ НА ДАННИТЕ

data = pd.read_csv('m1.csv', sep=',', decimal=".")

pdf = PdfPages("graphs_m1.pdf")

resultData = pd.DataFrame()

for q in range (99, 100):

    # Номер на ред с данни
    rowno = 269
    
    # Вземаме ред с данни
    row = data.iloc[rowno,7:].dropna()
    print("row\n", row)
    
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
    series_name = data.iloc[rowno,0]
    print("series_name:", series_name)
    
    # series_type ще съдържа типа на серията - годишна, месечна ... 
    series_type=data.iloc[rowno,4]
    print("series_type", series_type)
    
    # Някои проверки за валидност 
    assert datapoints + future_predictions == len(row.values)
    
    
    resultData.at[q, 'name']=series_name
    resultData.at[q,'datapoints']=datapoints
    resultData.at[q, 'future_predictions']=future_predictions
   
     
    fig = plt.figure(q, figsize=(32,18))
    plt.grid(True, dashes=(1,1))
    plt.title(series_name+series_type)
    plt.xticks(rotation=90)
    plt.plot(row, color="blue", label="Original data")
    plt.axvline(x=datapoints, color="red", linestyle="--", label="Forecast horizon")

    # Тройно експоненциално изглаждане
    if seasonality>1:
        holtwinters = TripleExponentialPredictionMethod(row, datapoints, numSeasons=seasonality)
    else:
        holtwinters = DoubleExponentialPredictionMethod(row, datapoints)
    prediction = holtwinters.predict()
    plt.plot(prediction, color="red", label="Exp. smoothing $\\alpha=%s$, $\\beta=%s$ $\\gamma=%s$, MAPE=%s, wMAPE=%s" % 
              (holtwinters.hwresults.params['smoothing_level'], holtwinters.hwresults.params['smoothing_trend'], 
               holtwinters.hwresults.params['smoothing_seasonal'], holtwinters.computeMAPE(), holtwinters.computeWMAPE()))
    plt.legend()

    resultData.at[q, 'holtwinters_mape']=holtwinters.computeMAPE()
    resultData.at[q, 'holtwinters_wmape']=holtwinters.computeWMAPE()

    # Проста RNN
    simpleRNN = SimpleRNNPredictionMethod(row, datapoints, numNeurons=(256,q,256))
    prediction = simpleRNN.predict()
    plt.plot(prediction, color="cyan", label="Simple RN, MAPE=%s, wMAPE=%s" % (simpleRNN.computeMAPE(), simpleRNN.computeWMAPE()))
    plt.legend()
    
    resultData.at[q, 'simplernn_mape']=simpleRNN.computeMAPE()
    resultData.at[q, 'simplernn_wmape']=simpleRNN.computeWMAPE()
    
    pdf.savefig(fig)

pdf.close()
resultData.to_csv('data.csv')