# -*- coding: utf-8 -*-
"""
Copyright 2024 Angelin Lalev

This program is free software: you can redistribute it and/or modify it under 
the terms of the GNU General Public License as published by the 
Free Software Foundation, either version 3 of the License, or 
(at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT 
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with 
this program. If not, see <https://www.gnu.org/licenses/>.
"""

"""
Data preparation for the GB models.
"""

import pandas as pd
import os
import sys
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import statsmodels.graphics.tsaplots as tsa
import random as rnd
from dateutil.relativedelta import relativedelta 
from scipy.stats import pearsonr


### CONSTANTS 

input_data_loc = os.path.join("transformed_data", "gbm_input-final.csv")

### INITIALIZATION


### LOADING OF THE DATA
data = pd.read_csv(input_data_loc, sep=',', decimal=".")

columns = data.columns.values
print(columns)
chosen = set(columns).difference(['Unnamed: 0' 'date' 'year_x' 'month_x' 'day' 'week' 'weekday' 'pollution'])
print(chosen)

cov = data.cov(numeric_only=True)['pollution']
print(cov)
cov.to_csv(os.path.join("results", "covmatrix.csv"))

for c in chosen:
    fig,ax=plt.subplots()
    ax.plot(data['pollution'], data[c], "ro")
    ax.set_ylabel(c)
    ax.set_title('pollution')
    plt.show()


