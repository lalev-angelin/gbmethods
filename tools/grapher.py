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
This program graphs various statistical plots of M3 competition
"""

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import altair as alt
import seaborn as sns
import numpy as np

original = pd.read_csv('../original_data/airpollution/LSTM-Multivariate_pollution.csv',sep=',', decimal=".")
original['date']=pd.to_datetime(original['date'])
original['month']=original['date'].dt.month
original['year']=original['date'].dt.year


# Plot

fig = plt.figure(figsize=(28,15))
plt.grid(True, dashes=(1,1))
plt.xticks(rotation=90)
plt.plot(original['pollution'], color="blue", label="Original data")
plt.legend()

# Aggregated dayly plot

data = original.copy(True)
data = data[['date', 'pollution']]
data['date'] = pd.to_datetime(data['date']).dt.date
aggregated  = (data.groupby(data['date']).mean())

fig = plt.figure(figsize=(28,15))
plt.grid(True, dashes=(1,1))
plt.xticks(rotation=90)
plt.plot(aggregated['pollution'], color="blue", label="Original data")
plt.legend()
plt.show()

# Aggregated monthly plot
aggregated = original[['date', 'year', 'month', 'pollution']]
print(aggregated)
aggregated = aggregated.groupby(['year', 'month']).mean()
aggregated['month'] = pd.to_datetime(aggregated['date']).dt.month
aggregated['year'] = pd.to_datetime(aggregated['date']).dt.year


fig = plt.figure(figsize=(28,15))
fig = sns.barplot(aggregated, x='date', y='pollution')
plt.xticks(rotation=90)
plt.show()


# Montly means
fig = sns.barplot(aggregated, x='month', y='pollution')
plt.show()
  
# Month by month over common meanc
fig, axis = plt.subplots(3, 4, figsize=(18,15), sharey=True)
mean = aggregated['pollution'].mean()
for m in range(1, 13):
    
    month = aggregated[aggregated['month']==m]
    print(month)
    axis[(m-1)//4, (m-1)%4].plot(month['year'], month['pollution'])
    axis[(m-1)//4, (m-1)%4].title.set_text(month.iloc[0,2])
    axis[(m-1)//4, (m-1)%4].axhline(y=mean, color='gray')
    axis[(m-1)//4, (m-1)%4].
plt.show()



