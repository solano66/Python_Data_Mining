'''
Collated by Ching-Shih Tsou 鄒慶士 博士 (Ph.D.) Distinguished Prof. at the Department of Mechanical Engineering/Director at the Center of Artificial Intelligence & Data Science (機械工程系特聘教授兼人工智慧暨資料科學研究中心主任), MCUT (明志科技大學); Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), NTUB (國立臺北商業大學)
Notes: This code is provided without warranty.
'''

#### 2.2.1_摘要統計量
import pandas as pd
dow30 = pd.read_csv('dow30.csv') # 資料集與程式碼在同一個目錄下，打開資料表看一下
dow30 = dow30.drop(dow30.columns[0], axis=1)

# 股價資料框結構
dow30.info() # Similar to str() in R. Without any missing values.
tmp = dow30.describe(include='all')

# 30支股票
dow30.symbol.value_counts()
dow30.symbol.value_counts().shape

# 全體平均數
# dow30['Open'].mean()
import numpy as np
np.mean(dow30.Open)

# 再一個例子
x = np.arange(1, 11)
np.mean(x)

# 小心遺缺值np.nan
xWithNan = np.hstack((x, np.nan)) # h: horizontal
np.mean(xWithNan)
np.nanmean(xWithNan)

# 截尾平均數
from scipy.stats import trim_mean
trim_mean(dow30['Open'], proportiontocut = 0.1)

# 中位數
np.median(x) # by numpy median() function

# 也稱為第50個(50th)百分位數
np.median(dow30['Open'])
dow30['Open'].median() # by pandas Series

# 水質樣本資料
algae = pd.read_csv("algae.csv") # make it from R by write.csv(algae, file="algae.csv", row.names = FALSE)

# 自訂眾數計算函數
def Mode(x):
    return pd.Series(x).value_counts().index[0]
    
Mode(algae['size'])

# 以次數分佈表核驗Mode函數計算結果
algae['size'].value_counts()

# pandas Series內建眾數計算函數
algae[['size']].mode()
algae['size'].mode()

# scipy.stats的眾數計算方法
# from scipy import stats
# stats.mode(algae['size'])

# scipy.stats的幾何平均數計算方法
from scipy import stats
stats.gmean(x)

# 最常用的四分位數加上最小與最大值
dow30.columns
dow30.Open.quantile([0, .25, .5, .75, 1]) # by pandas Series

# 自定義Tukey的五數摘要統計值 
# Tukey five number summary in Python
# https://stackoverflow.com/questions/3878245/tukey-five-number-summary-in-python
import numpy as np
from scipy.stats import scoreatpercentile
from numpy import nanmedian

def fivenum(v):
    """Returns Tukey's five number summary (minimum, lower-hinge, median, upper-hinge, maximum) for the input vector, a list or array of numbers based on 1.5 times the interquartile distance"""
    try:
        np.sum(v)
    except TypeError:
        print('Error: you must provide a list or array of only numbers')
    q1 = scoreatpercentile(v[~np.isnan(v)],25)
    q3 = scoreatpercentile(v[~np.isnan(v)],75)
    iqd = q3-q1
    md = nanmedian(v)
    whisker = 1.5*iqd
    return np.nanmin(v), md-whisker, md, md+whisker, np.nanmax(v) # md or 1st Q/3rd Q
    # return np.nanmin(v), q1, md, q3, np.nanmax(v)

fivenum(dow30.Open)

# How to Difference a Time Series Dataset with Python
# https://machinelearningmastery.com/difference-time-series-dataset-python/

# 全距
dow30.Open.max() - dow30.Open.min()

# 1到 10跨兩期的8個    (Why?)差分值                    
pd.Series(range(1, 11)).diff(periods=2)
# 二階差分值
pd.Series(range(1, 11)).diff(periods=2).diff()

# 四分位距
from scipy.stats import iqr
iqr(dow30.Open, axis=0)

# 變異數
import numpy as np
np.var(dow30.Open)
# 標準差
np.std(dow30.Open)

# 相較於其他語言：Matlab或R，numpy預設計算母體變異數與標準差(分母為n)，ddof=1可計算樣本變異數與標準差(分母為n-1)
np.std(dow30.Open, ddof=0)
np.std(dow30.Open, ddof=1)

# 直接由標準差與平均數計算變異係數
np.std(dow30.Open)/np.mean(dow30.Open)
np.std(range(1,11))/np.mean(range(1,11))

# 中位數絕對離差(mad)
import statsmodels.api as sm
sm.robust.scale.mad(dow30.Open)
dow30.Open.mad() # 千萬別搞混了！Pandas Series物件的mad方法是平均絕對離差(Return the 'mean' absolute deviation of the values for the requested axis)
from scipy import stats
stats.median_absolute_deviation(dow30.Open) # same as statsmodels (New in scipy version 1.3.0)

pd.DataFrame(abs(dow30.Open - dow30.Open.mean())).mean()
1.4826*pd.DataFrame(abs(dow30.Open - dow30.Open.median())).median() # Please check the documentation of R function mad(). Wow ! there is also a concept of low-median and high-median.

 # 衡量數據集不均與集結度的Python套件
# pip install inequality_coefficients
import inequality_coefficients as ineq
# 建立所得向量data
data = np.array([541, 1463, 2445, 3438, 4437, 5401, 6392, 8304, 11904, 22261])
# 吉尼集中度計算函數
gini_coeff = ineq.gini(data)
gini_coeff

# 自定義吉尼不純度函數
def Gini(x):
    return 1-np.sum((x.value_counts()/x.value_counts().sum())**2)

# 完美同質情況
Gini(pd.Series(["a","a","a","a","a","a"]))
# 非完美情況
Gini(pd.Series(["a","b","b","b","b","b"]))
# 完美異質情況
Gini(pd.Series(["a","b","c","d","e"]))
