'''
Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the IDS and CICD of NTUB (國立臺北商業大學資訊與決策科學研究所暨智能控制與決策研究室教授); at the ME Dept. and CAIDS of MCUT (2020~2022借調至明志科技大學機械工程系任特聘教授兼人工智慧暨資料科學研究中心主任); the CSQ (2019年起任中華民國品質學會大數據品質應用委員會主任委員); the DSBA (2013年創立臺灣資料科學與商業應用協會); and the CARS (2012年創立中華R軟體學會) 
Notes: This code is provided without warranty.
'''

#### A wiered example of histogram in R language R 語言的怪異例子
import pandas as pd
pd.Series([1,2,3,4,5]).hist()

series = pd.Series([1,2,3,4,5])

#### Get the binning details from numpy 從 numpy 取得分組的計算細節
import numpy as np
count, division = np.histogram(series)
#### Get the binning details from numpy 從 pandas 取得分組的計算細節
# count1, division1 = pd.np.histogram(series)

#### How about our assignment ? 我們的習題
algae = pd.read_csv('./data/algae.csv')
# By plot method from pandas Series 
algae.mxPH.hist()

#### Could you map the binning to the histogram ? 麻煩自己對應圖形
# count2, division2 = pd.np.histogram(algae.mxPH)
# ValueError: autodetected range of [nan, nan] is not finite
count2, division2 = np.histogram(algae.mxPH.dropna())

#### Let's check another variable "Cl" 另一個變量
algae.Cl.hist()
count3, division3 = np.histogram(algae.Cl.dropna())

#### Please go back to the wiered example in R 再回到第一個源自 R 語言的例子！
count, division = np.histogram(series, bins=4)
# Set bins to be division
series.hist(bins=division) # A similarly wiered example in Python ! 相似的怪！

#### Try numpy.histogram_bin_edges(,bins='sturges')
import numpy as np
np.histogram_bin_edges(algae.Cl , bins='sturges')
# ValueError: autodetected range of [nan, nan] is not finite

np.histogram_bin_edges(algae.Cl.dropna() , bins='sturges') # 10 bins for Cl

count_Cl, division_Cl = np.histogram(algae.Cl.dropna(), bins='auto') # 25 bins for Cl

import matplotlib.pyplot as plt
_ = plt.hist(algae.Cl.dropna(), bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram of Cl with 'auto' bins")
plt.show() # Please compare with the R histogram

count_mxPH, division_mxPH = np.histogram(algae.mxPH.dropna(), bins='auto') # 18 bins for mxPH

import matplotlib.pyplot as plt
_ = plt.hist(algae.mxPH.dropna(), bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram with 'auto' bins")
plt.show() # Please compare with the R histogram, there are nine bins returned from hist() in R.

#### Which one is the best language for AI&DS ? 到底哪一個是正港的 AI 統計計算語言？值得思考的問題。或許無需更換工具，不過建議多聽多看！