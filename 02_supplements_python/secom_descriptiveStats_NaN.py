'''
Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the IDS, CICD, and CISD of NTUB (臺北商業大學資訊與決策科學研究所暨智能控制與決策研究室教授兼校務永續發展中心主任); at the ME Dept. and CAIDS of MCUT (2020年借調至明志科技大學機械工程系擔任特聘教授兼人工智慧暨資料科學研究中心主任兩年); the CSQ (2019年起任中華民國品質學會大數據品質應用委員會主任委員); the DSBA (2013年創立臺灣資料科學與商業應用協會); and the CARS (2012年創立中華R軟體學會) 
Notes: This code is provided without warranty.
'''

import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

# Where –1 corresponds to a pass and 1 corresponds to a fail.
secom = pd.read_csv('uci-secom.csv') # (1567, 592) 去頭去尾後有590感測器訊號

# secom1 = pd.read_csv('secom.data', sep=' ', header=None) # (1567, 590)
# labels = pd.read_csv('secom_labels.data', sep=' ', header=None) # (1567, 2), time stamp included

# Python變量名稱建議不要用數字或/
secom.columns
secom.'0'
secom.0
secom['0']
secom[['0']]

# 變量名稱處理(兩個加號的中間是Python單列迴圈，[1:-1]去頭(0)去尾(-1, 但請注意前包後不包！))
secom.columns = ['Time'] + [''.join(['v',x]) for x in secom.columns[1:-1]] + ['PassFail']

secom.v0

secom['PassFail'].value_counts()

# 摘要統計表(類別變項看前四rows，數值變量看後七rows)
secom_sum = secom.describe(include='all')

secom.v5.value_counts(dropna=False) # NaN        14
secom.v5.isnull()
secom.v5.isnull().sum() # there are 14 Trues !

# 首先整張表辨識nan位置，並進行統計
secom.isnull().sum(axis=0).sort_values() # at most 1429 observations missing
secom.isnull().sum(axis=1).sort_values() # at most 152 variables missing

secom.isnull().sum(axis=1).value_counts(sort=False).sort_index() # 224, 239, 264, 170, ...

# 直接刪除
secom_naomit = secom.dropna(axis=0) # 居然半個不留！！！

# 590 * .2 = 118, 590 - 118 = 472

# 若樣本的完整變量個數有472以上(含)，則留之！
secom_over472 = secom.dropna(thresh=472)
secom_over472.isnull().sum(axis=1).sort_values()

# y 沒有遺缺值
secom.iloc[:,-1].isnull().sum()

# time stamp 沒有遺缺值
secom.iloc[:,0].isnull().sum()

# 全部樣本留下來，以各變項中位數填補遺缺值
cleanSecom = pd.DataFrame()
for col in secom.columns:
    if (col != 'Time'): # 排除時間戳記欄位
        cleanSecom[col] = secom[col].fillna(secom[col].median()) # 看程式碼有時要像學英語，請倒過來看！

# 重新加入時間戳記
cleanSecom = pd.concat([secom['Time'], cleanSecom], axis=1) # 其實是R語言的cbind()

cleanSecom.columns

# 確認是否已無遺缺值
cleanSecom.isnull().sum().sum() # 0 nans

secom.isnull().sum().sum() # 41951 nans

# 請就良品不良品，統計v1~v5的特徵中位數、算術平均數與標準差
secom.columns
secom.groupby('PassFail')

list(secom.groupby('PassFail'))

dir(secom.groupby('PassFail'))

secom.groupby('PassFail').ngroups # 2

secom.groupby('PassFail').groups # Sample index of each group

type(secom.groupby('PassFail').groups) # dict

# secom.groupby('PassFail').groups['1']

# agg({變量名：[統計值1, 統計值2,...]})
secom.groupby('PassFail').agg({"v1":['mean','median','std'], "v2":['mean','median','std'], "v3":['mean','median','std'], "v4":['mean','median','std'], "v5":['mean','median','std']})

import numpy as np
secom.groupby('PassFail').agg({"v1": [np.mean, np.median, np.std]})

secom.v4.mean()
secom.v4.median()
secom.v4.std()

secom[['v'+str(1),'PassFail']].groupby('PassFail').agg({'v'+str(1): [np.mean, np.median, np.std]})

#### 作業：完成所有變量的群組(y)摘要(sensors)分析。
secom.dtypes # 確認可用的摘要統計值/量數
summ = {} # An empty dict
for i in range(590):
    summ['v'+str(i)] = secom[['v'+str(i),'PassFail']].groupby('PassFail').agg({'v'+str(i): ['mean', 'median', 'std']}) # np.mean, np.median, np.std

summ['v234']
secom.v234.value_counts(dropna=False)

#### 作業：所有變量的群組(y)摘要(sensors)結果的視覺化分析。

