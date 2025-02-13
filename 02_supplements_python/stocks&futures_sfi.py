'''
Collated by Ching-Shih Tsou 鄒慶士 博士 (Ph.D.) Distinguished Prof. at the Department of Mechanical Engineering/Director at the Center of Artificial Intelligence & Data Science (機械工程系特聘教授兼人工智慧暨資料科學研究中心主任), MCUT (明志科技大學); Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), NTUB (國立臺北商業大學)
Notes: This code is provided without warranty.
'''

import os
print(os.getcwd())
os.chdir("D:\Jane\SCHOOL\金融資料")


import pandas as pd

# 載入csv檔
# stockdata = pd.read_csv("./股票/上市股票日資料/stock_data_2002.csv")
stockdata = pd.read_csv("./sfi_2019Aug13/stock_data_2002.csv")

# 檢視資料表前、後五筆數據
print(stockdata.head())
print(stockdata.tail())

stockdata.columns = ['opening', 'highest', 'lowest', 'closing', 'volume', 'date']
print(stockdata.head())

# 檢視資料表的維度與維數，7845筆觀測值，6個變數
print(stockdata.shape)

# 各變數或各觀測值遺缺狀況
print(stockdata.isnull().sum(axis=0))

print(stockdata.dtypes)
# 將date轉為時間格式
stockdata['date'] = pd.to_datetime(stockdata['date'])
print(stockdata['date'].head())
print(stockdata.info())

print(stockdata.columns)
print(stockdata['highest'].max())
print(stockdata['highest'].min())
print(stockdata.describe())

# 增加year欄位
stockdata['year'] = stockdata['date'].dt.year
# 增加month欄位
stockdata['month'] = stockdata['date'].dt.month

# 共31年份的資料
print(stockdata['year'].nunique())
# 2019年只有1月的資料
print(stockdata.groupby('year')['month'].nunique())
# 每年的最高開盤價與最低收盤價
print(stockdata.groupby('year')['opening'].max())
print(stockdata.groupby('year')['closing'].min())
# 每年成交量
print(stockdata.groupby('year')['volume'].sum())

# 2018年資料
stockdata_2018 = stockdata[stockdata['year'] == 2018]
# 2018年每月成交總量
print(stockdata_2018.groupby('month')['volume'].sum())


# find encoding of csv file：
#with open('./期貨/台指期貨tick資料/Daily_2017_09_01.csv') as f:
#   print(f)
futuresdata = pd.read_csv("./期貨/台指期貨tick資料/Daily_2017_09_01.csv", encoding = "cp950")

print(futuresdata.head())
print(futuresdata.tail())
print(futuresdata.shape)
print(futuresdata.info())
print(futuresdata.isnull().sum())
# 拿掉最後一筆NaN
futuresdata = futuresdata.iloc[:-1,:]
print(futuresdata.tail())

print(futuresdata['成交日期'].value_counts())
print(futuresdata['商品代號'].value_counts())
# 213種商品
print(futuresdata['商品代號'].nunique())

print(futuresdata.describe())

print(futuresdata.groupby('商品代號')['成交價格'].max())

df1 = futuresdata.groupby(['商品代號'])['成交價格'].max().reset_index(name='最高成交價')
df2 = futuresdata.groupby(['商品代號'])['成交價格'].min().reset_index(name='最低成交價')

result = pd.merge(df1, df2, on='商品代號')
# 最高成交價遞減排序
result1 = result.sort_values('最高成交價', ascending=False).iloc[:10,:]

import matplotlib.pyplot as plt
%matplotlib inline
fig = plt.figure()

plt.bar(result1['商品代號'], result1['最高成交價'], label = 'Highest price', align = "edge", width = 0.35)
plt.bar(result1['商品代號'], result1['最低成交價'], label = 'Lowest price', align = "edge", width = -0.35)
plt.legend()
plt.xlabel("Product code")
plt.ylabel("final price")
fig.savefig('futures_HighestTOP10.png')
