'''
Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the IDS, CICD, and CISD of NTUB (臺北商業大學資訊與決策科學研究所暨智能控制與決策研究室教授兼校務永續發展中心主任); at the ME Dept. and CAIDS of MCUT (2020年借調至明志科技大學機械工程系擔任特聘教授兼人工智慧暨資料科學研究中心主任兩年); the CSQ (2019年起任中華民國品質學會大數據品質應用委員會主任委員); the DSBA (2013年創立臺灣資料科學與商業應用協會); and the CARS (2012年創立中華R軟體學會) 
Notes: This code is provided without warranty.
'''

# In[1]:


# -*- coding: utf-8 -*-
import glob, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ## input file names

# In[2]:


input_files = glob.glob("./car ECU Datalogs/2018*.csv", recursive=True)
input_files


# ## read all data

# In[3]:


all_data = None
for i in range(len(input_files)):
    csv_columns = []
    csv_columns.append('time')
    header_number = 0
    channel = ""
    with open(input_files[i]) as f:
        lines = f.readlines()
        for line in lines:
            header_number += 1
            if line.startswith('Channel : '):
                channel = line.replace('Channel : ','').replace('\n','')
            if line.startswith('Type : '):
                csv_columns.append(channel + "[" + line.replace('Type : ','').replace('\n','') + "]")
            if line.startswith('Log : '):
                break
    print("{} header={}  file={}".format(i+1, header_number, input_files[i]))
    df = pd.read_csv(input_files[i], index_col=False, skiprows=header_number, names=csv_columns)
    
    basename = os.path.basename(input_files[i])
    df['date'] = basename.replace('.csv','').split('-')[0]
    route = basename.replace('.csv','').split('-')[1]
    df['route'] = route
    df['time'] = df['date'] + " " + df['time']
    df['time'] = pd.to_datetime( df['time'], format="%Y%m%d %H:%M:%S.%f")
    df['#time_diff'] = df['time'].diff(1).dt.total_seconds()
    df['#time_seq'] = df['#time_diff'].cumsum()
    df['#road_seq'] = df['#time_seq']
    if route == "mimos2home":
        df['#road_seq'] = df['#road_seq'].max() - df['#road_seq']

    if all_data is None:
        all_data = df
    else:
        all_data = pd.concat([all_data, df])
all_data


# In[4]:


all_data.info()


# ## Scatter plot

# In[5]:


plt.scatter(all_data['Load[Pressure]'], all_data['RPM[EngineSpeed]'], c=all_data['TargetAFR[AFR]'], cmap='Blues', s=5)
plt.colorbar()
plt.title("engine load and RPM")
plt.xlabel("Load[Pressure]")
plt.ylabel("RPM[EngineSpeed]")
plt.grid(True)


# In[6]:


plt.scatter(all_data['#time_seq'], all_data['Load[Pressure]'], c=all_data['TargetAFR[AFR]'], cmap='Blues', s=5)
plt.colorbar()
plt.title("engine load time series")
plt.xlabel("time_seq[sec]")
plt.ylabel("Load[Pressure]")
plt.grid(True)


# In[7]:


plt.scatter(all_data['#time_seq'], all_data['RPM[EngineSpeed]'], c=all_data['TargetAFR[AFR]'], cmap='Blues', s=5)
plt.colorbar()
plt.title("engine RPM time series")
plt.xlabel("time_seq[sec]")
plt.ylabel("RPM[EngineSpeed]")
plt.grid(True)

