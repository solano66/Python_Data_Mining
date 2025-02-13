#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from datetime import datetime
import time
from scipy.signal import find_peaks
from scipy.stats import gamma
import statsmodels.api as sm
import warnings


# In[3]:


train = pd.read_csv("./data/train_ML_IOT.csv")
test = pd.read_csv("./data/test_ML_IOT.csv")


# In[4]:

set(train.columns) - set(test.columns) # {'Vehicles'}

test_ID = test["ID"]
test.drop(["ID"],axis = 1,inplace=True)


# In[5]:


test['DateTime'] = pd.to_datetime(test['DateTime'])


# In[6]:


train['DateTime'] = pd.to_datetime(train['DateTime'])
train['Weekday'] = [datetime.weekday(date) for date in train.DateTime]
train['Year'] = [date.year for date in train.DateTime]
train['Month'] = [date.month for date in train.DateTime]
train['Day'] = [date.day for date in train.DateTime]
train['Hour'] = [date.hour for date in train.DateTime]
train['Week'] = [date.week for date in train.DateTime]
train['Quarter'] = [date.quarter for date in train.DateTime]
train["IsWeekend"] = train["Weekday"] >= 5 # 0: Sunday, 1: Monday, 2: Tuesday, 3: Wednesday, 4: Thursday, 5: Friday, 6: Saturday


test['DateTime'] = pd.to_datetime(test['DateTime'])
test['Weekday'] = [datetime.weekday(date) for date in test.DateTime]
test['Year'] = [date.year for date in test.DateTime]
test['Month'] = [date.month for date in test.DateTime]
test['Day'] = [date.day for date in test.DateTime]
test['Hour'] = [date.hour for date in test.DateTime]
test['Week'] = [date.week for date in test.DateTime]
test['Quarter'] = [date.quarter for date in test.DateTime]
test["IsWeekend"] = test["Weekday"] >= 5


# In[7]:

print(train.Junction.value_counts())
# 3    14592
# 2    14592
# 1    14592
# 4     4344

j1=train[train["Junction"]==1]
j2=train[train["Junction"]==2]
j3=train[train["Junction"]==3]
j4=train[train["Junction"]==4]

js = [j1, j2, j3, j4]


# In[8]:


plt.plot(np.diff(j1.Vehicles))


# # one hot encoding

# In[9]:


def datetounix(df):
    # Initialising unixtime list
    unixtime = []
    
    # Running a loop for converting Date to seconds
    for date in df['DateTime']:
        unixtime.append(time.mktime(date.timetuple()))
    
    # Replacing Date with unixtime list
    df['DateTime'] = unixtime
    return(df)


# In[10]:


train_feats = datetounix(train.drop(['Vehicles','Year', 'Quarter', 'Month'], axis=1))
test_feats = datetounix(test.drop(['Year', 'Quarter', 'Month'], axis=1))


# In[11]:


X = train_feats
X_test = test_feats


# In[12]:


X.Junction = X.Junction.astype("str")
X.Weekday = X.Weekday.astype("str")
X.Day = X.Day.astype("str")

X_test.Junction = X_test.Junction.astype("str")
X_test.Weekday = X_test.Weekday.astype("str")
X_test.Day = X_test.Day.astype("str")


# In[13]:


X = pd.get_dummies(X)
X_test = pd.get_dummies(X_test)
y = train.Vehicles.to_frame()


# In[14]:


X


# # Outlier Detection

# # prophet

# In[15]:


# get_ipython().system('pip install fbprophet')
from fbprophet import Prophet # !conda install -c conda-forge fbprophet --y


# In[16]:


def model_Prophet(j):
    j = j.rename(columns={'Vehicles': 'y', 'DateTime': 'ds'})
    
    train1 = j.sample(frac=0.9, random_state=25)
    test1 = j.drop(train1.index)
    
    m = Prophet(changepoint_range=0.95)
    m.fit(train1)
    
    future = m.make_future_dataframe(periods=119, freq='H') 
    forecast = m.predict(future)
    forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()
    result = pd.concat([j.set_index('ds')['y'], forecast.set_index('ds')[['yhat','yhat_lower','yhat_upper']]], axis=1)
    #fig1 = m.plot(forecast)
    
    result['error'] = result['y'] - result['yhat']
    result['uncertainty'] = result['yhat_upper'] - result['yhat_lower']
    result[result["error"].abs() > 1.5*result["uncertainty"]]
    result['anomaly'] = result.apply(lambda x: 'Yes' if(np.abs(x['error']) > 1.5*x['uncertainty']) else 'No', axis = 1)
    
    return result


# In[17]:


def plot_Prophet(result):
    fig = px.scatter(result.reset_index(), x='ds', y='y', color='anomaly', title='Vehicles')
    #slider
    fig.update_xaxes(
        rangeslider_visible = True,
        rangeselector = dict(
            buttons = list([
                dict(count=1, label='1y', step="year", stepmode="backward"),
                dict(count=2, label='3y', step="year", stepmode="backward"),
                dict(count=2, label='5y', step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    fig.show()


# In[18]:


for j in js:
    r = model_Prophet(j)
    plot_Prophet(r)


# # adtk
# Anomaly Detection Toolkit (ADTK) is a Python package for unsupervised / rule-based time series anomaly detection.

# As the nature of anomaly varies over different cases, a model may not work universally for all anomaly detection problems. Choosing and combining detection algorithms (detectors), feature engineering methods (transformers), and ensemble methods (aggregators) properly is the key to build an effective anomaly detection model.

# This package offers a set of common detectors, transformers and aggregators with unified APIs, as well as pipe classes that connect them together into a model. It also provides some functions to process and visualize time series and anomaly events.

# https://adtk.readthedocs.io/en/stable/

# In[19]:


# get_ipython().system('pip install adtk')
from adtk.visualization import plot # !conda install -c conda-forge adtk --y
from adtk.data import validate_series
from adtk.detector import SeasonalAD
from adtk.detector import AutoregressionAD
from adtk.detector import InterQuartileRangeAD
from adtk.detector import GeneralizedESDTestAD
from adtk.detector import PersistAD
from adtk.detector import LevelShiftAD
from adtk.detector import VolatilityShiftAD


# In[20]:


jads = [j1, j2, j3, j4]
for i,j in enumerate(js):
    jads[i] = j.Vehicles
    jads[i].index = j.DateTime


# In[21]:


seasonal_vol = SeasonalAD()
for jad in jads:
    jad = validate_series(jad)
    anomalies = seasonal_vol.fit_detect(jad)
    anomalies.value_counts()
    plot(jad, anomaly=anomalies, anomaly_color="orange", anomaly_tag="marker")
    plt.show()


# In[22]:


iqr_ad = InterQuartileRangeAD(c=1.5)
for jad in jads:
    anomalies = iqr_ad.fit_detect(jad)
    plot(jad, anomaly=anomalies, ts_linewidth=1, ts_markersize=3, anomaly_markersize=5, anomaly_color='red', anomaly_tag="marker");


# In[23]:


esd_ad = GeneralizedESDTestAD(alpha=0.3)
for jad in jads:
    anomalies = esd_ad.fit_detect(jad)
    plot(jad, anomaly=anomalies, ts_linewidth=1, ts_markersize=3, anomaly_markersize=5, anomaly_color='red', anomaly_tag="marker");


# In[24]:


persist_ad = PersistAD(c=8, side='positive')
for jad in jads:
    anomalies = persist_ad.fit_detect(jad)
    plot(jad, anomaly=anomalies, ts_linewidth=1, ts_markersize=3, anomaly_markersize=5, anomaly_color='red', anomaly_tag="marker");


# In[25]:


level_shift_ad = LevelShiftAD(c=3, side='both', window=5)
for jad in jads:
    anomalies = level_shift_ad.fit_detect(jad)
    plot(jad, anomaly=anomalies, ts_linewidth=1, ts_markersize=3, anomaly_markersize=5, anomaly_color='red', anomaly_tag="marker");


# In[26]:


volatility_shift_ad = VolatilityShiftAD(c=8, side='positive', window=30)
for jad in jads:
    anomalies = volatility_shift_ad.fit_detect(jad)
    plot(jad, anomaly=anomalies, ts_linewidth=1, ts_markersize=3, anomaly_markersize=5, anomaly_color='red', anomaly_tag="marker");


# # mean & median

# In[27]:


sns.set(rc={'figure.figsize':(16, 4)})


# In[28]:


def model_mean(j, std_param = 3, window_length = 720):
    
    rmeans = j.rolling(window_length, min_periods=1).mean()
    rstd = j.rolling(window_length, min_periods=1).std()
    
    idx1 = (j>(rmeans+std_param*rstd))
    idx2 = (j<(rmeans-std_param*rstd))
    idx = idx1 | idx2
    
    upper = rmeans+std_param*rstd
    lower = rmeans-std_param*rstd
    
    return j, idx, rmeans, upper, lower


# In[29]:


def plot_med_mean(j, idx, rmeans, upper, lower, std_param = 3, alpha = 0.015):
    plt.plot(j)
    plt.plot(rmeans, color = 'red')
    plt.plot(upper, color = 'orange')
    plt.plot(lower, color = 'orange')
    plt.xlabel("Date")
    plt.ylabel("Vehicles")
    plt.scatter(j.index[idx],j[idx],color = 'purple',zorder=10)
    plt.show()


# In[30]:


for j in jads:
    j, idx, rmeans, upper, lower = model_mean(j)
    plot_med_mean(j, idx, rmeans, upper, lower)


# In[31]:


def model_median(j, std_param = 3, window_length = 720):
    
    rmeans = j.rolling(window_length, min_periods=1).median()
    rstd = j.rolling(window_length, min_periods=1).std()
    
    idx1 = (j>(rmeans+std_param*rstd))
    idx2 = (j<(rmeans-std_param*rstd))
    idx = idx1 | idx2
    
    upper = rmeans+std_param*rstd
    lower = rmeans-std_param*rstd
    
    return j, idx, rmeans, upper, lower


# In[32]:


for j in jads:
    j, idx, rmeans, upper, lower = model_median(j)
    plot_med_mean(j, idx, rmeans, upper, lower)


# In[33]:


def model_alp_mean(j, std_param = 3, window_length = 720, alpha = 0.015):
    
    rmeans = j.rolling(window_length, min_periods=1).mean()
    rstd = j.rolling(window_length, min_periods=1).std()
    
    idx1 = (j>(rmeans+std_param*rstd*np.exp(-alpha*rstd)))
    idx2 = (j<(rmeans-std_param*rstd*np.exp(-alpha*rstd)))
    idx = idx1 | idx2
    
    upper = rmeans+std_param*rstd*np.exp(-alpha*rstd)
    lower = rmeans-std_param*rstd*np.exp(-alpha*rstd)
    
    return j, idx, rmeans, upper, lower


# In[34]:


for j in jads:
    j, idx, rmeans, upper, lower = model_alp_mean(j)
    plot_med_mean(j, idx, rmeans, upper, lower)


# In[35]:


def model_alp_median(j, std_param = 3, window_length = 720, alpha = 0.015):
    
    rmeans = j.rolling(window_length, min_periods=1).median()
    rstd = j.rolling(window_length, min_periods=1).std()
    
    idx1 = (j>(rmeans+std_param*rstd*np.exp(-alpha*rstd)))
    idx2 = (j<(rmeans-std_param*rstd*np.exp(-alpha*rstd)))
    idx = idx1 | idx2
    
    upper = rmeans+std_param*rstd*np.exp(-alpha*rstd)
    lower = rmeans-std_param*rstd*np.exp(-alpha*rstd)
    
    return j, idx, rmeans, upper, lower


# In[36]:


for j in jads:
    j, idx, rmeans, upper, lower = model_alp_median(j)
    plot_med_mean(j, idx, rmeans, upper, lower)


# In[37]:


def model_kurt_mean(j, std_param = 3, window_length = 720, alpha = 0.015):
    
    rmeans = j.rolling(window_length, min_periods=1).median()
    rstd = j.rolling(window_length, min_periods=1).std()
    rstd_lower = j[j<rmeans].rolling(window_length ,min_periods=1).std()
    
    kurt = j.kurt()
    
    upper = rmeans+std_param*rstd*np.exp(0.04*kurt)
    lower = rmeans - (5*rstd_lower)    
    idx1 = (j>upper)
    idx2 = (j<lower)
    
    idx = idx1 | idx2
    
    return j, idx, rmeans, upper, lower


# In[38]:


for j in jads:
    j, idx, rmeans, upper, lower = model_kurt_mean(j)
    plot_med_mean(j, idx, rmeans, upper, lower)
    j[idx] = np.nan


# In[39]:


j1 = j1.interpolate(method="ffill")
j2 = j2.interpolate(method="ffill")
j3 = j3.interpolate(method="ffill")
j4 = j4.interpolate(method="ffill")


# In[40]:


j4['Vehicles'].isnull().values.any()

