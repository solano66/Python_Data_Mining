'''
Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the IDS, CICD, and CISD of NTUB (臺北商業大學資訊與決策科學研究所暨智能控制與決策研究室教授兼校務永續發展中心主任); at the ME Dept. and CAIDS of MCUT (2020年借調至明志科技大學機械工程系擔任特聘教授兼人工智慧暨資料科學研究中心主任兩年); the CSQ (2019年起任中華民國品質學會大數據品質應用委員會主任委員); the DSBA (2013年創立臺灣資料科學與商業應用協會); and the CARS (2012年創立中華R軟體學會) 
Notes: This code is provided without warranty.
'''

#Sys.setenv(RETICULATE_PYTHON = "/opt/anaconda3/bin/python")

## ----setup, include=FALSE--------------------------------------
#library(reticulate)
#use_python("/opt/anaconda3/bin")
#use_python("/opt/anaconda3/bin/python3")
#use_python("/opt/anaconda3/lib/python3.7")


####  *A.1.2 資結的變裝秀_alternate*
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

df = pd.read_csv('./data/tiressus.csv')
# 車速逐步上升的輪胎耐久性測試資料(Stepped Up Speed Tire Failure Test Data)。
# 資料來源：https://www.nhtsa.gov/equipment/tires

# df是衍生資料結構：'pandas DataFrame'。
print(type(df))

# 以dir()函數查看'pandas DataFrame'擁有的屬性和方法。
print(dir(df))
[(nam, type(getattr(df, nam))) for nam in dir(df)]
[(a, 'func' if callable(getattr(df, a)) else 'attr') for a in dir(df)]


# df.dtypes的輸出(各變量類型)是衍生資料結構：'pandas Series'。
print(df.dtypes)
# Unnamed: 0                      int64
# Phase                           int64
# Tire_Type                      object
# Barcode                         int64
# Dot_Number                     object
# Dot_MidWeekDate                object
# Collection_Date                object
# DOT_Age                       float64
# X1st_Task                      object
# X1st_Task_Status               object
# Position                       object
# ORN                            object
# AZ_Use                          int64
# DOT_Est_Mileage_mi              int64
# DOT_Est_Mileage_km              int64
# Initial_IP_kPa                  int64
# Load_kg                       float64
# Time_To_Failure               float64
# Speed_At_Failure_km_h           int64
# Mileage_At_Failure_km           int64
# Millions_Cycles_At_Failure    float64
# Failure_Type                   object
# Failure.Notes                  object
# Photo_1                        object
# Photo_2                        object
# Photo_3                        object
# Photo_4                        object
# Invoice_Date                   object
# dtype: object

print(type(df.dtypes))


# 過長的變量名稱重新命名。
df = df.drop(['Unnamed: 0'], axis=1)
df.rename(columns = {'Millions_Cycles_At_Failure': 'Cycles_At_Failure', 'Failure.Notes': 'Failure_Notes'}, inplace = True)
print(df.columns)


# DataFrame的columns屬性是衍生資料結構：'pandas Index'。
print(type(df.columns))


# DataFrame的columns屬性轉為原生資料結構('to_list'方法)：'list'。
print(df.columns.to_list())
print(len(df.columns.to_list()))
print(type(df.columns.to_list()))

# df.DOT_Age取出單欄(兩種取法)後是衍生資料結構：'pandas Series'。
print(df.DOT_Age.head())
print(type(df['DOT_Age']))

# df[['DOT_Age']]轉成原生資料結構字典，'DOT_Age'是鍵(key)，值(value)是變量值。

dict(df[['DOT_Age']])

# 鍵'DOT_Age'下有衍生資料結構pandas Series。
dict(df[['DOT_Age']])['DOT_Age'].head()
type(dict(df[['DOT_Age']])['DOT_Age']) # pandas.core.series.Series

#### *C.1 機率分佈簡介*
#### Python伯努利機率質量尖峰圖(spike plot)
import matplotlib.pyplot as plt
plt.style.use('ggplot') # Grammar of Graphics 圖形文法

import seaborn as sns # !conda install anaconda::seaborn --y
# 設定seaborn繪圖美學參數，color_code設為True時會將簡記的顏色碼重新映射到seaborn色盤
sns.set(color_codes=True)
# settings for seaborn plot sizes
sns.set(rc={'figure.figsize':(5,5)})

from scipy.stats import bernoulli
data_bern = bernoulli.rvs(p=0.6,size=10000) # rvs: Random Variable Samples

import numpy as np
np.unique(data_bern, return_counts = True)

ax = sns.distplot(data_bern,
                  kde=False,
                  hist_kws={'alpha':1}) # distplot: DISTribution PLOT
ax.set(xlabel='Bernoulli Distribution', ylabel='Frequency')
plt.show()

# Probability Distributions in Python (https://www.datacamp.com/community/tutorials/probability-distributions-python)
# Get data points from Seaborn distplot (https://stackoverflow.com/questions/37374983/get-data-points-from-seaborn-distplot)

#### Python二項式機率質量尖峰圖(spike plot)
from scipy.stats import binom
data_binom = binom.rvs(n=10,p=0.8,size=10000)

np.unique(data_binom, return_counts = True)

ax = sns.distplot(data_binom,
                  kde=False,
                  hist_kws={'alpha':1})
ax.set(xlabel='Binomial Distribution', ylabel='Frequency')
plt.show()

#### Python卜桑/波氏Poisson機率質量尖峰圖(spike plot)
from scipy.stats import poisson
data_poisson = poisson.rvs(mu=3, size=10000)

np.unique(data_poisson, return_counts = True)

ax = sns.distplot(data_poisson,
                  bins=30,
                  kde=False,
                  hist_kws={'alpha':1})
ax.set(xlabel='Poisson Distribution', ylabel='Frequency')
plt.show()

#### Python負二項分佈機率值量尖峰圖(spike plot)
from scipy.stats import nbinom
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)
# nbinom takes n and p as shape parameters where n is the number of successes, p is the probability of a single success, and 1−p is the probability of a single failure.
n, p = 5, 0.4
mean, var, skew, kurt = nbinom.stats(n, p, moments='mvsk')

x = np.arange(nbinom.ppf(0.01, n, p),
              nbinom.ppf(0.99, n, p))
ax.plot(x, nbinom.pmf(x, n, p), 'bo', ms=8, label='nbinom pmf')
ax.vlines(x, 0, nbinom.pmf(x, n, p), colors='b', lw=5, alpha=0.5)
plt.show()

#### Python連續型均勻密度曲線
# import uniform distribution
from scipy.stats import uniform

# random numbers from uniform distribution
n = 10000
start = 10
width = 20
data_uniform = uniform.rvs(size=n, loc = start, scale = width) # loc: 從何處開始; scale: 延展多寬

ax = sns.distplot(data_uniform,
                  bins=100,
                  kde=True,
                  hist_kws={'alpha':1})
ax.set(xlabel='Uniform Distribution ', ylabel='Frequency')
plt.show()

#### Python指數密度曲線
from scipy.stats import expon
data_expon = expon.rvs(scale=1,loc=0,size=1000)

ax = sns.distplot(data_expon,
                  kde=True,
                  bins=100,
                  hist_kws={'alpha':1})
ax.set(xlabel='Exponential Distribution', ylabel='Frequency')
plt.show()

#### Python常態密度曲線
import matplotlib.pyplot as plt

import numpy as np

import scipy.stats as stats

import math

mu = 0

variance = 1

n = 25

sigma = math.sqrt(variance)

x = np.linspace(-3, 3, 100) # 產生隨機變數值

plt.plot(x, stats.norm.pdf(x, mu, sigma)) # 引用stats下常態分佈pdf計算函數，並以matplotlib繪製曲線圖

plt.show()

#### Python常態累積分佈曲線
stats.norm.cdf((0))

import scipy.stats as stats

stats.norm.cdf(-1)

stats.norm.cdf(1.96) - stats.norm.cdf(-1.96)

#### Python伽碼密度曲線
from scipy.stats import gamma
data_gamma = gamma.rvs(a=5, size=10000)

ax = sns.distplot(data_gamma,
                  kde=True,
                  bins=100,
                  hist_kws={'alpha':1})
ax.set(xlabel='Gamma Distribution', ylabel='Frequency')
plt.show()

#### *C.2 抽樣分佈與中央極限定理*
#### 中央極限定理(Central Limit Theorem, CLT)

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

mu = 0
variance = 1
n = 25
sigma = math.sqrt(variance/n)
#x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
x = np.linspace(-3, 3, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma))
n = 5
sigma = math.sqrt(variance/n)
plt.plot(x, stats.norm.pdf(x, mu, sigma))
n = 1
sigma = math.sqrt(variance/n)
plt.plot(x, stats.norm.pdf(x, mu, sigma))
plt.xlim((-3, 3))
plt.show()

#### ***時間序列資料之繪圖探索案例(after CLT)(change to astsa)


#### *C.3 估計與假說檢定*
#### 平均數檢定
#### 單一常態母體Z檢定
# statsmodels套件下的$Z$檢定，其檢定用標準差是用估計的，因此本例只是示範而已，其實樣本不夠大，我們用樣本標準差$s$估計母體標準差$\sigma^{2}$，並且使用標準常態$Z$作為抽樣分佈其實是不客觀的！。

from statsmodels.stats.weightstats import ztest

BAC = [79.5, 80.1, 80.6, 80.8, 82.4]
stat_t, p_z = ztest(BAC, value = 80, alternative = 'larger')

stat_t
p_z

import numpy as np
np.array(BAC).mean()
np.array(BAC).std()
np.array(BAC).std(ddof=1)

# p-value = 0.08沒有足夠的證據顯示BAC的平均水平大於80。

#### 單一常態母體，小樣本，$\sigma^{2}$未知$t$檢定(右尾)
BAC = [79.5, 80.1, 80.6, 80.8, 82.4]

stat_t = (np.array(BAC).mean() - 80)/(np.array(BAC).std(ddof=1)/np.sqrt(len(BAC)))

stat_t

from scipy.stats import t
df = len(BAC) - 1
p_t = t.sf(stat_t, df)

p_t

# p-value = 0.12沒有足夠的證據顯示BAC的平均水平大於80。

#### 單一樣本$t$檢定儀器校正例子(雙尾檢定)
import pandas as pd
RI_1 = pd.read_csv('./data/RI_1.txt', sep = '\t')

stat_t = (RI_1.mean() - 1.52214)/(RI_1.std()/np.sqrt(RI_1.shape[0]))

stat_t

from scipy.stats import t
df = RI_1.shape[0] - 1
p_t = t.sf(stat_t, df)

2*p_t # 雙尾p值

#### 單一樣本$t$檢定(tiressus.csv)
import pandas as pd
tiressus = pd.read_csv('./data/tiressus.csv')

tiressus.dtypes

con = (tiressus.Tire_Type=="H") & (tiressus.Speed_At_Failure_km_h==160)

typeh = tiressus.loc[(tiressus.Tire_Type=="H") & (tiressus.Speed_At_Failure_km_h==160),'Time_To_Failure']

typeh.mean()

from scipy import stats
t, p = stats.ttest_1samp(typeh, popmean = 9)

#### 兩獨立樣本$t$檢定，常態母體，變異數均未知，但假設它們相等
from scipy.stats import t
2 * t.sf(4.01,8)

#### 變異數比較檢定
#### F檢定(常態樣本，雙樣本)
import pandas as pd
field_goals = pd.read_csv('./data/fieldgoals.csv', index_col=[0]) # 指定索引欄
field_goals.info()

import re

new_columns = []
for element in field_goals.columns:
    new_columns.append(re.sub('[\.]', '_', str(element)))

field_goals.columns = new_columns

field_goals.head()

field_goals['outside'] = field_goals['stadium_type'] == 'Out' # 正是R的field.goals.inout資料框

# good = field_goals.loc[field_goals.play_type=='FG good', ['yards', 'stadium_type', 'outside']]

# good.reset_index(drop=True, inplace=True)

# bad = field_goals.loc[field_goals.play_type=='FG no', ['yards', 'stadium_type', 'outside']]

# bad.reset_index(drop=True, inplace=True)

# t.test(yards~outside, data=good)
# t.test(yards~outside, data=bad)

# field_goals.inout <- transform(field_goals, outside=(stadium.type=='Out'))
# str(field_goals.inout)
# t.test(yards~outside, data=field_goals.inout)

# mean(field_goals.inout$yards[field_goals.inout$outside == TRUE])
# mean(field_goals.inout$yards[field_goals.inout$outside == FALSE])

field_goals.columns

X = field_goals.loc[field_goals.outside == False, 'yards']
Y = field_goals.loc[field_goals.outside == True, 'yards']

F = X.var() / Y.var()
print("檢定統計量的值是{}".format(F))

df1 = len(X) - 1; df2 = len(Y) - 1
print("分子與分母的自由度分別是{}與{}".format(df1, df2))

alpha = 0.05 #Or whatever you want your alpha to be.
print("顯著水準是{}".format(alpha))

from scipy import stats

# 右尾
p_value = stats.f.sf(F, df1, df2)
#p_value = 1 - stats.f.cdf(F, df1, df2)
print("右尾檢定的p值是{}".format(p_value))

# 雙尾
p_value = 2 * stats.f.sf(F, df1, df2)
print("雙尾檢定的p值是{}".format(p_value))

if p_value > alpha:
  print("Do not reject the null hypothesis that true ratio of variances is equal to 1")
else:
  print("Reject the null hypothesis that true ratio of variances is equal to 1")

#### ***Bartlett's test(after F test)(vis_alpha_cat.feather)(change to R in a Nutshell)


#### Levene’s test(change to R in a Nutshell)


#### Fligner-Killeen's test(change to R in a Nutshell)


#### Normality test(change to R in a Nutshell)


#### D’Agostino’s K^2 Test(change to R in a Nutshell)


#### Anderson-Darling Test(change to R in a Nutshell)


#### 常態性檢定(change to R in a Nutshell)
#### Shapiro-Wilk常態性檢定(change to R in a Nutshell)


#### D’Agostino’s K^2檢定(change to R in a Nutshell)


#### Anderson-Darling檢定(change to R in a Nutshell)


#### *Jarque-Bera test (比較skewness and kurtosis. cf. SDAFE p.86)(change to R in a Nutshell)
# Note that this test only works for a large enough number of data samples (>2000) as the test statistic asymptotically has a Chi-squared distribution with 2 degrees of freedom.


#### Hotelling's T^2 scores in python (https://stackoverflow.com/questions/25412954/hotellings-t2-scores-in-python)


#### ***相關性檢定(july_2017_v2.csv)(change to R in a Nutshell)


#### Pearson’s Correlation Coefficient(change to R in a Nutshell)


# In this case when you have a large sample size, the probability to reject the null hypothesis is greater, then the result is that you have a significant p-value with a very low correlation coefficient.
# STATISTICAL SIGNIFICANCE IN BIG DATA(https://bigdataeconometrics.wordpress.com/2013/12/28/statistical-significance-in-big-data/)


#### Spearman’s Rank Correlation (algae.csv)
algae = pd.read_csv('./data/algae.csv')
algae.info()
algae.isnull().sum()

# Example of the Spearman's Rank Correlation Test
from scipy.stats import spearmanr

stat, p = spearmanr(algae.mxPH, algae.NO3, nan_policy='omit') # default is ‘propagate’, this returns nan
print('spearmanr=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')

#### Kendall’s Rank Correlation
# Example of the Kendall's Rank Correlation Test
from scipy.stats import kendalltau

stat, p = kendalltau(algae.mnO2, algae.PO4, nan_policy='omit')
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')


#### Chi-Squared Test

# Example of the Chi-Squared Test
import pandas as pd
from scipy.stats import chi2_contingency
table = pd.crosstab(algae['size'], algae['speed']) # try algae.size !
stat, p, dof, expected = chi2_contingency(table)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')

#### *TODO: Kolmogorov–Smirnov test

#### 時間序列檢定
#### 硬碟故障資料集
import feather
hd = feather.read_dataframe('./data/SmartData_2017_ST12000NM0007.feather')
hd.iloc[:,:11].head()
hd.info()

hd.describe(include='all')

hd.serial_number.value_counts() # ZCH05LYL頻率最高

hd.loc[hd.serial_number == 'ZCH05LYL', :].isnull().sum()

# 1: Read error rate
# *5: Reallocated sectors count(全是0)
# *9: Power-on hours
# 10: Spin retry count
# x184: End-to-end error/IOEDC('ZCH05LYL'無)
# *187: Reported uncorrectable errors
# 188:Command timeout
# 190, 192~195
# x196: Reallocation event count ('ZCH05LYL'無)
# *197: Current pending sector count (全是0，與5負相關)
# 198: (Offline) Uncorrectable sector count (與197完全相同)
# x201: Soft read error rate or TA counter detected ('ZCH05LYL'無)


#### 自我相關函數與相關圖 ACF (or correlogram)
from statsmodels.graphics import tsaplots

smart1r = hd.loc[hd.serial_number == 'ZCH05LYL', 'smart_1_raw'] # 116筆
smart5r = hd.loc[hd.serial_number == 'ZCH05LYL', 'smart_5_raw'] # 116筆
smart9r = hd.loc[hd.serial_number == 'ZCH05LYL', 'smart_9_raw'] # 116筆
smart10r = hd.loc[hd.serial_number == 'ZCH05LYL', 'smart_10_raw'] # 116筆
smart188r = hd.loc[hd.serial_number == 'ZCH05LYL', 'smart_188_raw'] # 116筆
smart197r = hd.loc[hd.serial_number == 'ZCH05LYL', 'smart_197_raw'] # 116筆

# np.arange(len(corr))

tsaplots.plot_acf(smart1r)

#### 偏自我相關函數 PACF
# A partial autocorrelation is a summary of the relationship between an observation in a time series with observations at prior time steps with the relationships of intervening observations removed.
tsaplots.plot_pacf(smart1r)

# Reference: A Gentle Introduction to Autocorrelation and Partial Autocorrelation (https://machinelearningmastery.com/gentle-introduction-autocorrelation-partial-autocorrelation/)

#### 交叉相關函數 CCF: Finding Lagged Correlations Between Two Time Series
import matplotlib.pyplot as plt
fig = plt.figure()

ax1 = fig.add_subplot(311)
ax1.plot(smart1r)
ax1.set_xlabel('Read error rate')

ax2 = fig.add_subplot(312)
ax2.plot(smart9r)
ax2.set_xlabel('Power-on hours')

ax3 = fig.add_subplot(313)
ax3.set_xlabel('Cross correlation of Read error rate and power-on hours')
ax3.xcorr(smart1r, smart9r, usevlines=True, maxlags=None, normed=True, lw=2) # usevlines (setting it to True), we can instruct matplotlib to use vlines() instead of plot() to draw the lines of the correlation plot.
ax3.grid(True)
plt.ylim(-1, 1)
plt.tight_layout()
plt.show()

fig = plt.figure()
plt.xcorr(smart1r, smart9r, usevlines=False, maxlags=None, normed=True, marker=',')
plt.show() # 2 * 116 - 1 = 231


# Python時間序列檢驗方法大多在statsmodels套件下的tsa模組

#### *Box-Pierce檢定 (自我相關的Box test)(加example)
# The Box-Pierce test is a simplified version of the Ljung-Box test.
# Let n = length(x), rhoi = autocorrelation of x at lag i, k = lag, then the Box-Pierce test statistic is n * (rho1^2 + rho2^2 + ... + rhok^2) and the Ljung-Box test statistic is n*(n+2)*(rho1^2/(n-1) + rho2^2/(n-2) + ... + rhok^2/(n-k) Under the null hypothesis of no autocorrelation, the test statistics have a Chi-square distribution with lag degrees of freedom.

# Reference: Box-Pierce and Ljung-Box Tests (https://docs.tibco.com/pub/enterprise-runtime-for-R/4.0.2/doc/html/Language_Reference/stats/Box.test.html)

#### *Ljung-Box檢定 (AR(p)的Box test)(加example)
# Also known as Portmanteau test or Simultaneous test

#### Augmented Dickey-Fuller (ADF) Unit Root檢定 (均數復歸的時間序列test)
# Testing for Mean Reversion
# 檢驗一時間序列樣本是否有單根，亦即有趨勢或是自我迴歸性
# You want to know if your time series is mean reverting.
# When a time series is mean reverting, it tends to return to its long-run average. It may wander off, but eventually it wanders back. If a time series is not mean reverting, then it can wander away without ever returning to the mean.
# 背景假設：

# 觀測值依時序排列

# 欲檢定的假說：

# H0: 有單根(序列非穩態).(No evidence of mean reverting)
# H1: 無單根(序列是穩態)(The time series is likely mean reverting)

# Example of the Augmented Dickey-Fuller unit root test
from statsmodels.tsa.stattools import adfuller
#hd.columns
data = hd.loc[hd.serial_number == 'ZCH05LYL', 'smart_1_raw'] # 116筆
stat, p, lags, obs, crit, t = adfuller(data)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably not Stationary')
else:
	print('Probably Stationary')

data.plot()


#### *Phillips-Perron檢定 (另一種單根檢定)(加example)


#### Kwiatkowski-Phillips-Schmidt-Shin (KPSS)檢定 (第三種單根檢定)
# 檢驗一時間序列樣本是否是趨勢穩態

# 背景假設：

# 觀測值依時序排列

# 欲檢定的假說：

# H0: 時間序列非趨勢穩態.
# H1: 時間序列是趨勢穩態.

# Example of the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test
from statsmodels.tsa.stattools import kpss

stat, p, lags, crit = kpss(data)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably not Stationary')
else:
	print('Probably Stationary')

# 與adfuller結果不相同

# 還有偏相關係數

#### *C.5 迴歸分析簡介*(加example)

#### References: 17 Statistical Hypothesis Tests in Python (Cheat Sheet)
# https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/