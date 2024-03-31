import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from OurJSONEncoder import OurJSONEncoder
import json
from random import randint
import seaborn as sns
import numpy as np


### TUNING AND CONFIG

# The locations of the input and output files
transformed_data_loc = os.path.join("transformed_data", "gbm_input-final.csv")
result_dir = os.path.join("results")


# To avoid NA values in the specified columns
skip_first_n_rows = 48

# To avoid NA values in the specified columns
skip_last_n_rows = 12

# LGBMRegressor parameters
gbmparams = {"num_leaves":31,  
             "max_depth":-1,
             "learning_rate":0.1,
             "n_estimators":1000}

# Where will we stop the training and take measure
test_periods = [x for x in range(24, 2400, 24)]



### SUBROUTINES

# Computes MAPE of two sequences
def computeMAPE(actual, forecast): 
    assert len(actual)==len(forecast)

    mape = 0
    for i in range(0, len(actual)):
       mape = mape + abs((actual[i]-forecast[i])/actual[i])

    
    return mape/len(actual)

# Computes Weighted MAPE of two sequences
def computeWMAPE (actual, forecast):
    assert len(actual)==len(forecast)

    sumdiff = 0
    div = 0
    for i in range(0, len(actual)):
        sumdiff = sumdiff + abs(actual[i]-forecast[i])
        div = div + abs(actual[i])
    
    return sumdiff/div

def computeSMAPE (actual, forecast): 
    assert len(actual)==len(forecast)

    sumdiff=0
    for i in range(0, len(actual)):
        nomin = abs(forecast[i]-actual[i])
        denomin = (abs(actual[i])+abs(forecast[i]))/2
        sumdiff=sumdiff + (nomin/denomin)

    return (sumdiff * 100)/len(actual)


### RELEVANT COLUMNS
### |Corrleation coeff.| >0.2 

columns = ['wnd_spd', "wnd_spd-1", "wnd_spd-2", "wnd_spd-3", "wnd_spd-4", "wnd_spd-5", 
           "wnd_spd-6", "wnd_spd-7", "wnd_spd-8", "wnd_spd-9",
           "pollution-1", "pollution-2", "pollution-3", "pollution-4", "pollution-5", 
           "pollution-6", "pollution-7", "pollution-8", "pollution-9", "pollution-10", 
           "pollution-11", "pollution-12", "pollution-13", "pollution-14", "pollution-15", 
           "pollution-16", "pollution-17", "pollution-18", "pollution-19", "pollution-20", 
           "pollution-21", "pollution-22",  "pollution-23", "d_avg_pollution-1"]



### IMPORT DATA
data = pd.read_csv(transformed_data_loc, sep=',', decimal='.')


for h in range (0, 12):
    if h!=0:
        pol_col_name='pollution+%d'%h
    else:
        pol_col_name='pollution'

    ### SPLIT 
    
    X = data[columns][skip_first_n_rows:-skip_last_n_rows].reset_index().drop('index', axis=1)
    Y = data[pol_col_name][skip_first_n_rows:-skip_last_n_rows].reset_index().drop('index', axis=1)
    dates = data['date'][skip_first_n_rows:-skip_last_n_rows].reset_index().drop('index', axis=1)
    
    
    trainX=X[:-24]
    trainY=Y[:-24]
    #print(trainX)
    #print(trainY)
    
    testX=X[-24:]
    testY=Y[-24:]
    #print(testX)
    #print(testY)
    
    ### FIT
    regressor = LGBMRegressor(boosting_type="gbdt", 
                              num_leaves=gbmparams['num_leaves'],
                              max_depth=gbmparams['max_depth'], 
                              learning_rate=gbmparams['learning_rate'], 
                              n_estimators=gbmparams['n_estimators'])
    
    regressor.fit(trainX, trainY)
    
        
    ### PREDICT
    predictY = pd.DataFrame(regressor.predict(X))
    print(predictY)
    
    ### PLOT
    residuals = pd.concat([dates['date'], Y, predictY], axis=1)
    residuals.rename(columns={residuals.columns.values[2]: "projected_pollution_lightgbm"}, inplace=True)
    residuals['residuals']=residuals[pol_col_name] - residuals['projected_pollution_lightgbm']
    print(residuals)
    
    
    
    sys.exit(0)
    
    
    
    
    
    fig = plt.figure() 
#    sns.lineplot(x=residuals['date'][-72:], y=residuals[pol_col_name][-72:])
#    sns.lineplot(x=residuals['date'][-72:], y=residuals['projected_pollution_lightgbm'][-72:])
    sns.lineplot(x=range(0,72), y=residuals[pol_col_name][-72:], label='actual', linestyle="dotted")
    sns.lineplot(x=range(0,72), y=residuals['projected_pollution_lightgbm'][-72:], label='projected')
    plt.xlabel("Pollution on: %s + x hours"%residuals['date'].iloc[-72])   
    plt.axvline(72-24, color="red", linestyle='--')
#    ax.xticks(rotation=90)
    plt.show()

    
    fig = plt.figure(figsize=(19,6)) 
    sns.lineplot(x=residuals['date'][-72:], y=residuals['residuals'][-72:])
    plt.axvline(72-24, color="red", linestyle='--')
    plt.xticks(rotation=90)
    plt.show()
    
   
    ### SAVE
    
    lst_data = residuals[pol_col_name].tolist()
    lst_projection = residuals['projected_pollution_lightgbm'].tolist()
    
    saveData = {}
    saveData['method']='LightGBM'
    saveData['parameters']=gbmparams
    saveData['testpoints']=test_periods
    saveData['data']=lst_data
    saveData['projection']=lst_projection
    saveData['sMAPE']=computeSMAPE(lst_data, lst_projection)
    strout = json.dumps(saveData, indent=2, cls=OurJSONEncoder)
    
    file = open(os.path.join(result_dir, "lightgbm.json"), "w")
    file.write(strout)
    file.close()