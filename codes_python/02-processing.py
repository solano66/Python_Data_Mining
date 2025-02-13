'''
Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the IDS, CICD, and CISD of NTUB (臺北商業大學資訊與決策科學研究所暨智能控制與決策研究室教授兼校務永續發展中心主任); at the ME Dept. and CAIDS of MCUT (2020年借調至明志科技大學機械工程系擔任特聘教授兼人工智慧暨資料科學研究中心主任兩年); the CSQ (2019年起任中華民國品質學會大數據品質應用委員會主任委員); the DSBA (2013年創立臺灣資料科學與商業應用協會); and the CARS (2012年創立中華R軟體學會)
Notes: This code is provided without warranty.
'''

#### 2.1.2 Python語言資料排序
## ------------------------------------------------------------------------
import pandas as pd
# 印出pandas 套件版次
print(pd.__version__)
USArrests = pd.read_csv("./_data/USArrests.csv")
# 檢視讀檔結果，注意奇怪欄名(Unnamed: 0)！
print(USArrests.head())

## ------------------------------------------------------------------------
# 修正欄位名稱
# USArrests.rename(columns={'Unnamed: 0':'state'}, inplace=True) # 條條大路通羅馬！
USArrests.columns = ['state', 'Murder', 'Assault',
'UrbanPop', 'Rape'] # 後四欄的欄位名重抄一遍，較不聰明！
# 設定state 為索引(index 從上面流水號變成下面州名)
USArrests = USArrests.set_index('state')
# Python 檢視資料表前五筆數據，類似R 語言head(USArrests)
print(USArrests.head())
# Python 檢視資料表的維度與維數(shape)
print(USArrests.shape)

## ------------------------------------------------------------------------
# 預設是依橫向第一軸(axis = 0) 的索引名稱升冪(ascending) 排列
print(USArrests.sort_index().head())
# 可調整為依縱向第二軸(axis = 1) 的索引名稱降冪(descending) 排列
print(USArrests.sort_index(axis=1, ascending = False).head())
# 依Rape 欄位值，沿第一軸(axis = 0) 降冪排列
print(USArrests.sort_values(by="Rape", ascending=False).
head())
# 也可以依兩欄位排序，前面欄位值平手時用後面欄位值排序
print(USArrests.sort_values(by=["Rape","UrbanPop"],
ascending=False).head())

# sort_values()平手狀況理解
# USArrests.sort_values(by=["UrbanPop","Rape"], ascending=False) # 特別留意Arizona, Florida, Texas與Utah的UrbanPop名次(如何tie-breaking)

USArrests.loc[:,['UrbanPop',"Rape"]].sort_values(by=["UrbanPop","Rape"], ascending=False)


## ------------------------------------------------------------------------
# 沿第二軸(axis = 1) 同一觀測值的四項事實數據名次
print(USArrests.rank(axis=1, ascending=False).head())
# 各欄位沿第一軸(axis = 0) 的五十州排名值
print(USArrests.rank(axis=0, ascending=False).head())

# rank()平手狀況理解(method='average')
# USArrests.rank(axis=0, ascending=False) # 特別留意Arizona, Florida, Texas與Utah的UrbanPop名次均為10.5((9, 10, 11, 12) -> 10.5, (9+10+11+12)/4=10.5)
# tmp1 = USArrests.loc[:,['UrbanPop',"Rape"]].rank(axis=0, ascending=False)

# 同名時取最大名次值(method 預設為average)
print(USArrests.rank(axis=0, ascending=False,
method="max")[:10])

# rank()平手狀況理解(method='max')
# tmp2 = USArrests.rank(axis=0, ascending=False, method="max") # (9, 10, 11, 12) -> 12

#### 2.1.4 Python語言資料變形
## ------------------------------------------------------------------------
USArrests = pd.read_csv("./_data/USArrests.csv")
# 變數名稱調整
USArrests.columns = ['state', 'Murder', 'Assault',
'UrbanPop', 'Rape']
# pandas 寬表轉長表(Python 語法中句點有特殊意義，故改為底線'_')
USArrests_dfl = (pd.melt(USArrests, id_vars=['state'],
var_name='fact', value_name='figure'))
print(USArrests_dfl.head())
# pandas 長表轉寬表
# index 為橫向變數，columns 為縱向變數，value 為交叉值
print(USArrests_dfl.pivot(index='state', columns='fact',
values='figure').head())

#### 2.1.6 Python語言資料清理
## ------------------------------------------------------------------------
algae = pd.read_csv("./_data/algae.csv")
# 單變量遺缺值檢查
# R 語言語法可想成是head(isnull(algae['mxPH'])))
print(algae['mxPH'].isnull().head())
# 注意Python 輸出格式化語法({} 搭配format() 方法)
print(" 遺缺{}筆觀測值".format(algae['mxPH'].isnull().sum()))

## ------------------------------------------------------------------------
# 利用pandas 序列方法dropna() 移除單變量遺缺值
mxPH_naomit = algae['mxPH'].dropna()
print(len(mxPH_naomit))
# 檢視整個資料表的遺缺狀況
print(algae.isnull().iloc[45:55,:5])
# 橫向移除不完整的觀測值(200 筆移除16 筆)
algae_naomit = algae.dropna(axis=0)
print(algae_naomit.shape)

## ------------------------------------------------------------------------
# 以thresh 引數設定最低變數個數門檻(200 筆移除9 筆)
algae_over17 = algae.dropna(thresh=17)
print(algae_over17.shape)

## ------------------------------------------------------------------------
# 各變數遺缺狀況：Chla 遺缺觀測值數量最多, Cl 次之...
algae_nac = algae.isnull().sum(axis=0)
print(algae_nac)
# 各觀測值遺缺狀況：遺缺變數個數
algae_nar = algae.isnull().sum(axis=1)
print(algae_nar[60:65])

## ------------------------------------------------------------------------
# 檢視不完整的觀測值(algae_nar>0 回傳橫向遺缺數量大於0 的樣本)
print(algae[algae_nar > 0][['mxPH', 'mnO2', 'Cl', 'NO3',
'NH4', 'oPO4', 'PO4', 'Chla']])
# 遺缺變數個數大於0(i.e. 不完整) 的觀測值編號
print(algae[algae_nar > 0].index)
# 不完整的觀測值筆數
print(len(algae[algae_nar > 0].index))
# 檢視遺缺變數超過變數個數algae.shape[1] 之20% 的觀測值
print(algae[algae_nar > algae.shape[1]*.2][['mxPH', 'mnO2',
'Cl', 'NO3', 'NH4', 'oPO4', 'PO4', 'Chla']])
# 如何獲取上表的橫向索引值？
print(algae[algae_nar > algae.shape[1]*.2].index)

## ------------------------------------------------------------------------
# 以drop() 方法，給IndexRange，橫向移除遺缺嚴重的觀測值
algae=algae.drop(algae[algae_nar > algae.shape[1]*.2].index)
print(algae.shape)

#### 2.2.3 Python語言群組與摘要
## ------------------------------------------------------------------------
# 載入必要套件
import pandas as pd
import numpy as np
import dateutil
# 載入csv 檔
path = '.'
fname = '/_data/phone_data.csv'
data = pd.read_csv(''.join([path, fname])) # index_col = [0]
# 830 筆觀測值，7 個變數
print(data.shape)
# 除index 與duration 外，所有欄位都是字串型別的類別變數
print(data.dtypes)
# 從編號1 的第2 欄位向後選，去除index 欄位
data = data.iloc[:,1:]
print(data.head())

## ------------------------------------------------------------------------
# 將日期字串逐一轉為時間格式(pandas Series物件的apply()隱式迴圈方法)
data['date'] = data['date'].apply(dateutil.parser.parse,
dayfirst=True)
# 也可以運用pandas 的to_datetime() 方法
data['date'] = pd.to_datetime(data['date']) # dayfirst=False, 請看說明文件的Warnings
# 'date' 的資料型別已改變
print(data.dtypes)

## ------------------------------------------------------------------------
# 傳入原生串列物件創建pandas 序列，index 引數給定橫向索引
series = pd.Series([20, 21, 12], index=['London',
'New York','Helsinki'])
print(series)
# pandas 序列物件apply() 方法的多種用法
# 可套用內建函數，例如：對數函數np.log()
print(series.apply(np.log))

# 也可以套用關鍵字為lambda 的匿名函數
# 其x 代表序列物件的各個元素
print(series.apply(lambda x: x**2))

# 或是自定義函數square()
def square(x):
    return x**2

print(series.apply(square))

# 另一個自定義函數，請注意參數custom_value 如何傳入
def subtract_custom_value(x, custom_value):
    return x - custom_value

# 以args 引數傳入值組(5,) 作為custom_value 參數
print(series.apply(subtract_custom_value, args=(5,)))

## ------------------------------------------------------------------------
# 檢視變數名稱(或是data.keys())
print(data.columns)
## Index(['index', 'date', 'duration', 'item', 'month',
## 'network', 'network_type'], dtype='object')

## ------------------------------------------------------------------------
# 服務類型次數分佈
print(data['item'].value_counts())
# 網路服務型式次數分佈
print(data['network_type'].value_counts())

## ------------------------------------------------------------------------
# 語音/數據最長服務時間
print(data['duration'].max())
# 語音通話的總時間計算，邏輯值索引+ 加總方法sum()
print(data['duration'][data['item'] == 'call'].sum())
# 每月記錄筆數
print(data['month'].value_counts())

## ------------------------------------------------------------------------
# 網路營運商家數
print(data['network'].nunique())
# 網路營運商次數分佈表
print(data['network'].value_counts())

## ------------------------------------------------------------------------
# 各欄位遺缺值統計
print(data.isnull().sum())

## ------------------------------------------------------------------------
# 依月份分組，先轉為串列後僅顯示最後一個月的分組數據
print(list(data.groupby(['month']))[-1])
# 分組數據是pandas 資料框的groupby 類型物件
print(type(data.groupby(['month'])))
# groupby 類型物件的groups 屬性是字典結構
print(type(data.groupby(['month']).groups))
# 以年-月為各組數據的鍵，觀測值索引為值
print(data.groupby(['month']).groups.keys())
## dict_keys(['2014-11', '2014-12', '2015-01', '2015-02',
## '2015-03'])
# '2015-03' 該組數據長度
print(len(data.groupby(['month']).groups['2015-03']))
# 取出'2015-03' 該組101 筆數據的觀測值索引
print(data.groupby(['month']).groups['2015-03'])
# first() 方法取出各月第一筆資料，可發現各組數據欄位與原數據相同
print(data.groupby('month').first())

## ------------------------------------------------------------------------
# 各月電信服務總時數(Christmas 前很忙！)
print(data.groupby('month')['duration'].sum())

## ------------------------------------------------------------------------
# 各電信營運商語音通話總和
print(data[data['item'] == 'call'].groupby('network')
['duration'].sum())

## ------------------------------------------------------------------------
# 多個欄位分組，各月各服務類型的資料筆數統計
# 抓出分組數據的任何欄位統計筆數均可，此處以date 為例
print(data.groupby(['month', 'item'])['date'].count())

## ------------------------------------------------------------------------
# 分組統計結果pandas 序列(duration 變數名稱在最下面)
print(data.groupby('month')['duration'].sum())
print(type(data.groupby('month')['duration'].sum()))
# 分組統計結果pandas 資料框(duration 變數名稱在上方)
print(data.groupby('month')[['duration']].sum())
print(type(data.groupby('month')[['duration']].sum()))

## ------------------------------------------------------------------------
# 群組數據後的agg() 分組統計
print(data.groupby('month').agg({"duration": "sum"}))
# 分組索引值為分組變數month 的值
print(data.groupby('month').agg({"duration": "sum"}).index)
## Index(['2014-11', '2014-12', '2015-01', '2015-02',
## '2015-03'], dtype='object', name='month')
# 分組統計的欄位名稱為duration
print(data.groupby('month').agg({"duration": "sum"}).columns)
# as_index=False 改變預設設定，month 從索引變成變數
print(data.groupby('month', as_index=False).agg({"duration":
"sum"}))

## ------------------------------------------------------------------------
# 各組多個統計計算
# 各月(month) 各服務(item) 的服務時間、網路服務型式與日期統計
print(data.groupby(['month', 'item']).agg({'duration':
[min, max, sum], 'network_type': "count", 'date':
[min, 'first', 'nunique']}))
#                 duration                        network_type
#                      min        max         sum        count
# month     item
# 2014-11   call     1.000   1940.000   25547.000          107
#           data    34.429     34.429     998.441           29
#           sms      1.000      1.000      94.000           94
# 2014-12   call     2.000   2120.000   13561.000           79
#           data    34.429     34.429    1032.870           30
#           sms      1.000      1.000      48.000           48
# 2015-01   call     2.000   1859.000   17070.000           88
#           data    34.429     34.429    1067.299           31
#           sms      1.000      1.000      86.000           86
# 2015-02   call     1.000   1863.000   14416.000           67
#           data    34.429     34.429    1067.299           31
#           sms      1.000      1.000      39.000           39
# 2015-03   call     2.000  10528.000   21727.000           47
#           data    34.429     34.429     998.441           29
#           sms      1.000      1.000      25.000           25
#
#
#                               date
# month     item                 min   first           nunique
# 2014-11   call 2014-10-15 06:58:00   2014-10-15 06:58:00 104
#           data 2014-10-15 06:58:00   2014-10-15 06:58:00 29
#           sms  2014-10-16 22:18:00   2014-10-16 22:18:00 79
# 2014-12   call 2014-11-14 17:24:00   2014-11-14 17:24:00 76
#           data 2014-11-13 06:58:00   2014-11-13 06:58:00 30
#           sms  2014-11-14 17:28:00   2014-11-14 17:28:00 41
# 2015-01   call 2014-12-15 20:03:00   2014-12-15 20:03:00 84
#           data 2014-12-13 06:58:00   2014-12-13 06:58:00 31
#           sms  2014-12-15 19:56:00   2014-12-15 19:56:00 58
# 2015-02   call 2015-01-15 10:36:00   2015-01-15 10:36:00 67
#           data 2015-01-13 06:58:00   2015-01-13 06:58:00 31
#           sms  2015-01-15 12:23:00   2015-01-15 12:23:00 27
# 2015-03   call 2015-02-12 20:15:00   2015-02-12 20:15:00 47
#           data 2015-02-13 06:58:00   2015-02-13 06:58:00 29
#           sms  2015-02-19 18:46:00   2015-02-19 18:46:00 17

# 請分析不同行動通訊服務下，服務時間(duration)的平均值、中位數及標準差，並說明可能的發現。
data.groupby(['item']).agg({'duration':[np.mean, np.median, np.std]})


####        2.3.2.1 奇異值矩陣分解
## ------------------------------------------------------------------------
# 載入陣列與奇異值分解類別
from numpy import array
from scipy.linalg import svd
A = array([[1, 2], [3, 4], [5, 6]])
print(A)
# 奇異值矩陣分解，輸出U, s, VT 三個方陣或矩陣
U, s, VT = svd(A)
print(U)
# 稍後以s 中的兩個值產生3*2 對角矩陣
print(s)
print(VT)

## ------------------------------------------------------------------------
# numpy 套件與矩陣代數密切相關
from numpy import diag
# 點積運算方法
from numpy import dot
# 零值矩陣創建
from numpy import zeros
# 創建m*n 階Sigma 矩陣，預存值為零
Sigma = zeros((A.shape[0], A.shape[1]))
# 對Sigma 矩陣植入2*2 對角方陣
Sigma[:A.shape[1], :A.shape[1]] = diag(s)
print(Sigma)
# 點積運算重構原矩陣
B = U.dot(Sigma.dot(VT))
print(B)

## ------------------------------------------------------------------------
# 3*3 方陣
A = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(A)
# SVD 方陣分解
U, s, VT = svd(A)
print(U)
# 中間的Sigma 亦為方陣
Sigma = diag(s)
print(Sigma)
print(VT)
# 點積運算重構原矩陣
B = U.dot(Sigma.dot(VT))
print(B)

## ------------------------------------------------------------------------
# 3*10 矩陣
A = array([
[1,2,3,4,5,6,7,8,9,10],
[11,12,13,14,15,16,17,18,19,20],
[21,22,23,24,25,26,27,28,29,30]])
# SVD 分解
U, s, VT = svd(A)
# 創建m*n 階矩陣，預存值為零
Sigma = zeros((A.shape[0], A.shape[1]))
# 對Sigma 矩陣植入對角方陣
Sigma[:A.shape[0], :A.shape[0]] = diag(s)
# 以前兩個最大的奇異值做SVD 近似計算
n_elements = 2
Sigma = Sigma[:, :n_elements]
VT = VT[:n_elements, :]
# 計算近似矩陣B((3*3).(3*2).(2*10))
B = U.dot(Sigma.dot(VT))
print(B)

## ------------------------------------------------------------------------
# SVD 降維運算((3*3) * (3*2))
T = U.dot(Sigma)
print(T)
# 另一種SVD 降維運算方式((3*10).(10*2))
T = A.dot(VT.T)
print(T)

## ------------------------------------------------------------------------
# 載入sklearn 的SVD 分解降維運算類別
from sklearn.decomposition import TruncatedSVD
# 3*10 矩陣
A = array([
[1,2,3,4,5,6,7,8,9,10],
[11,12,13,14,15,16,17,18,19,20],
[21,22,23,24,25,26,27,28,29,30]])
# 宣告SVD 分解降維空模
svd = TruncatedSVD(n_components=2)
# 配適實模
svd.fit(A)
# 轉換應用
result = svd.transform(A)
print(result)

