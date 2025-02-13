'''
Collated by Ching-Shih Tsou 鄒慶士 博士 (Ph.D.) Distinguished Prof. at the Department of Mechanical Engineering/Director at the Center of Artificial Intelligence & Data Science (機械工程系特聘教授兼人工智慧暨資料科學研究中心主任), MCUT (明志科技大學); Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), NTUB (國立臺北商業大學); the Chinese Academy of R Software (CARS) (中華R軟體學會創會理事長) and the Data Science and Business Applications Association of Taiwan (DSBA) (臺灣資料科學與商業應用協會創會理事長)
Notes: This code is provided without warranty.
'''

#### 基於訂單交易價值的RFM客戶價值度細分
# > 案例背景
# 
# * 銷售型公司基於訂單交易，發展客戶價值模型，以調整營運需求。
# * 客戶的狀態是動態變化的。
# * 定期更新以兼顧歷史狀態變化，例如：每週。
# * 模型結果一部分直接做營運分析，一部分要寫回資料庫中，作為其他數據建模的基本數據維度。

import time  # 導入時間庫/套件/包(package)
import numpy as np  # 導入numpy庫
import pandas as pd  # 導入pandas庫
# import mysql.connector  # 導入mysql連接庫


# 讀取數據(包含日期、客戶、金額等)
dtypes = {'ORDERDATE': object, 'ORDERID': object, 'AMOUNTINFO': np.float32}  # 設置各縱行數據類型
raw_data = pd.read_csv('./sales.csv', dtype=dtypes, index_col='USERID')  # 讀取數據文件，留意客戶編號USERID設為index

# raw_data.columns
# raw_data.index.value_counts()

#### > 數據審查和校驗
# 
# - 客戶編號(index)、訂購日期、訂單編號(以上類別變量)與訂單金額。
# - 主要分析數值變數價量訊息，依其分佈特性：樣本數、均值、極值、標準差、分位數等，做後續計算的輔助參考。
# - 注意訂單金額最小值($0.5)。

# 數據概覽
print ('Data Overview:')
print (raw_data.dtypes) # 列印原始數據各欄型別
print ('-' * 30)
print (raw_data.head(4))  # 列印原始數據前4條/筆樣本
print ('-' * 30)
print ('Data DESC:')
print (raw_data.describe(include='all'))  # 列印原始數據基本描述性信息(AMOUNTINFO明顯右偏：mean >> median)
print ('-' * 60)


# - 欄位ORDERDATE與AMOUNTINFO有遺缺情形(不足86135)。
# - 十筆紀錄有遺缺情形。

# 遺缺值審查
na_cols = raw_data.isnull().any(axis=0)  # 查看每一縱行是否具有遺缺值
print ('NA Cols:')
print (na_cols)  # 查看具有遺缺值的縱行
print ('-' * 30)
na_lines = raw_data.isnull().any(axis=1)  # 查看每一橫列是否具有遺缺值
print ('NA Recors:')
print ('Total number of NA lines is: {0}'.format(na_lines.sum()))  # 查看具有遺缺值的橫列總記錄數
print (raw_data[na_lines])  # 只查看具有遺缺值的橫列信息
print ('-' * 60)


#### > 遺缺資料、異常處理和格式轉換
#
# 遺缺與異常值處理
sales_data = raw_data.dropna()  # 丟棄帶有遺缺值的橫列記錄(86135 -> 86125)
sales_data = sales_data[sales_data['AMOUNTINFO'] > 1]  # 丟棄訂單金額<=1的記錄(86125 -> 84342)


# 日期格式轉換
sales_data['ORDERDATE'] = pd.to_datetime(sales_data['ORDERDATE'], format='%Y-%m-%d')  # 將字符串轉換為日期格式
print ('Raw Dtypes:')
print (sales_data.dtypes)  # 列印輸出數據框所有縱行的數據類型(留意datetime64[ns])
print ('-' * 60)


#### > RFM計算
# 
# - 59676名客戶購買記錄。
# - 各客戶**分組**後統計最近一次購買時間、資料期間購買次數、資料期間購買總金額

# 數據群組與摘要(依客戶群組後摘要日期與金額相關數據)
print(sales_data.index)
print(pd.Series(sales_data.index).value_counts().shape) # 59676名客戶購買記錄
recency_value = sales_data['ORDERDATE'].groupby(sales_data.index).max()  # 計算原始最近一次訂單時間(群組後用max()摘要'ORDERDATE')
frequency_value = sales_data['ORDERDATE'].groupby(sales_data.index).count()  # 計算原始訂單頻率(群組後用count()摘要'ORDERDATE')
monetary_value = sales_data['AMOUNTINFO'].groupby(sales_data.index).sum()  # 計算原始訂單總金額(群組後用sum()摘要'AMOUNTINFO')


# - 資料期間為2016一整年：2016-01-01 00:00:00到2016-12-29 00:00:00

sales_data.describe(include='all')


# - 設定參考時間點後計算最近購買日數Recency。
# - RFM分級計分。
# - R原始數值越大離指定日期越遠，所以分級分數越低。

#### 計算RFM得分
# 分別計算R、F、M得分
deadline_date = pd.to_datetime('2017-01-01')  # 指定一個時間節點，用於計算其他時間與該時間的距離
r_interval = (deadline_date - recency_value).dt.days  # 計算R間隔

# RFM分級計分
r_score = pd.cut(r_interval, 5, labels=[5, 4, 3, 2, 1])  # 計算R得分
f_score = pd.cut(frequency_value, 5, labels=[1, 2, 3, 4, 5])  # 計算F得分
m_score = pd.cut(monetary_value, 5, labels=[1, 2, 3, 4, 5])  # 計算M得分


#### - 分級計分後合併為資料框。

# R、F、M數據合併
rfm_list = [r_score, f_score, m_score]  # 將r、f、m三個維度組成列表
rfm_cols = ['r_score', 'f_score', 'm_score']  # 設置r、f、m三個維度縱行名
rfm_pd = pd.DataFrame(np.array(rfm_list).transpose(), dtype=np.int32, columns=rfm_cols,
                      index=frequency_value.index)  # 建立r、f、m數據框
print ('RFM Score Overview:')
print (rfm_pd.head(4))
print ('-' * 60)


#### - 計算總分與擴增欄位。

# 計算RFM總得分
# 方法一：加權得分(r: 0.6, f: 0.3, m: 0.1)
rfm_pd['rfm_wscore'] = rfm_pd['r_score'] * 0.6 + rfm_pd['f_score'] * 0.3 + rfm_pd['m_score'] * 0.1
# 方法二：RFM組合
rfm_pd_tmp = rfm_pd.copy()
rfm_pd_tmp['r_score'] = rfm_pd_tmp['r_score'].astype('str')
rfm_pd_tmp['f_score'] = rfm_pd_tmp['f_score'].astype('str')
rfm_pd_tmp['m_score'] = rfm_pd_tmp['m_score'].astype('str')
rfm_pd['rfm_comb'] = rfm_pd_tmp['r_score'].str.cat(rfm_pd_tmp['f_score']).str.cat(rfm_pd_tmp['m_score'])


# - 注意各項級分與加權總分的最大與最小值。

# 列印輸出和儲存結果
# 列印結果
print ('Final RFM Scores Overview:')
print (rfm_pd.head(4))  # 列印數據前4項結果
print ('-' * 30)
print ('Final RFM Scores DESC:')
print (rfm_pd.describe())


# - 結果存出。

# 儲存RFM得分到本地文件
rfm_pd.to_csv('/Users/Vince/cstsouMac/Python/Examples/MachineLearning/data/sales_rfm_score.csv')  # 儲存數據為csv


# ## 參考文獻 Reference: 
# 
# - Raschka, S. (2015), Python Machine Learning: Unlock deeper insights into machine learning with this vital guide to cutting-dege predictive analytics, PACKT Publishing.
# - 宋天龍 (2018), Python數據分析與數據化化運營, 機械工業出版社.
