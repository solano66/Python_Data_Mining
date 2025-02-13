'''
Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the IDS, CICD, and CISD of NTUB (臺北商業大學資訊與決策科學研究所暨智能控制與決策研究室教授兼校務永續發展中心主任); at the ME Dept. and CAIDS of MCUT (2020年借調至明志科技大學機械工程系擔任特聘教授兼人工智慧暨資料科學研究中心主任兩年); the CSQ (2019年起任中華民國品質學會大數據品質應用委員會主任委員); the DSBA (2013年創立臺灣資料科學與商業應用協會); and the CARS (2012年創立中華R軟體學會)
Notes: This code is provided without warranty.
'''

#### 2.1.5 R語言資料清理
import numpy as np
x = [1, 2, 3, np.NaN]
# 向量元素加總產生nan
y = x[0] + x[1] + x[2] + x[3]
y

# 加總函數的結果也是NA
z = np.sum(x)
z

# 移除NA後再做加總計算
z = np.nansum(x)
z

# pandas Series遺缺值NA辨識函數
import pandas as pd
pd.Series(x).isnull() # Similar to is.na() in R

# 取得遺缺值位置/樣本編號(Which one is TRUE?)
np.where(pd.Series(x).isnull()) # Similar to which() in R

#### Data cleaning on missing values
import pandas as pd
from io import StringIO

csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,,,8.0
0.0,11.0,12.0,'''

csv_data
type(csv_data) # str

StringIO(csv_data)
type(StringIO(csv_data)) # _io.StringIO

df = pd.read_csv(StringIO(csv_data))
df

help(df.isnull)

df.isnull()
df.isnull().sum() # from variable perspective
df.values # np.array behind

df.dropna()
df.dropna(axis=1)
df.dropna(how='all') # default 'any'

df.dropna(thresh=4) # drop rows that have not at least 4 non-NaN values

df.dropna(subset=['C']) # drop rows that column 'C' has NaN

#### Imputation by scikit-learn
import sklearn
sklearn.__version__ # '0.23.1' (from v0.20.4 to v0.22.2 is the grace period)
#from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer
#imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr = imr.fit(df)
imputed_data = imr.transform(df.values) # input numpy.ndarray
imputed_data

imputed_data = imr.transform(df) # you can also input pandas.DataFrame
imputed_data # same result

#### Case study: algae data set (page 164)
#### 2.1.6 Python語言資料清理
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
algae = pd.read_csv("algae.csv")

pd.set_option('display.max_row', 500)
pd.set_option('display.max_column', 500)

#### 各變量摘要統計表整合在資料表下方
algae_summary = algae.describe(include='all')
algae_new = pd.concat([algae, algae_summary])

#### 單變量遺缺值檢查
# R 語言語法可想成是head(isnull(algae['mxPH'])))
print(algae['mxPH'].isnull().head())

nr_nan = algae['mxPH'].isnull().sum()

# 注意Python 輸出格式化語法({} 搭配format() 方法)
print(" 遺缺 {} 筆觀測值".format(nr_nan))

## ------------------------------------------------------------------------
# 利用pandas 序列方法dropna() 移除單變量遺缺值
mxPH_naomit = algae['mxPH'].dropna()
print(len(mxPH_naomit))

#### 檢視整個資料表的遺缺狀況
print(algae.isnull().iloc[45:55,:5]) # is.na(df) -> df.isnull()
# 橫向移除不完整的觀測值(200 筆移除16 筆)
algae_naomit = algae.dropna(axis=0)
print(algae_naomit.shape)

#### 移除遺缺程度嚴重的樣本
# 以thresh 引數設定最低變數個數門檻(200 筆移除2 筆: 61和198)
algae_over13 = algae.dropna(thresh=13)
print(algae_over13.shape)

#### 縱橫統計遺缺樣貌
# 各變數遺缺狀況：Chla 遺缺觀測值數量最多, Cl 次之...
algae_nac = algae.isnull().sum(axis=0) # column sum, 勿忘合成函數sum(is.na(algae)) or apply(is.na(algae), 2, sum) in R
print(algae_nac)
# 各觀測值遺缺狀況：遺缺變數個數
algae_nar = algae.isnull().sum(axis=1) # row sum
print(algae_nar[60:65])

#### 檢視不完整的觀測值
# algae_nar>0 回傳橫向遺缺數量大於0 的樣本
print(algae[algae_nar > 0])
# 遺缺變數個數大於0(i.e. 不完整) 的觀測值編號
print(algae[algae_nar > 0].index)

# 檢視遺缺變數超過變數個數algae.shape[1] 之20% 的觀測值
print(algae[algae_nar > algae.shape[1]*.2])
# 如何獲取上表的橫向索引值？
print(algae[algae_nar > algae.shape[1]*.2].index)

## ------------------------------------------------------------------------
# 以drop() 方法，給IndexRange，橫向移除遺缺嚴重的觀測值
algae=algae.drop(algae[algae_nar > algae.shape[1]*.2].index)
print(algae.shape)

#### 以下為2.1.6 Python語言資料清理補充代碼(填補方式)
#### mxPH單一補值
mxPH = algae['mxPH'].dropna() # 須先移除NaNs後再繪圖或計算
#fig, ax = plt.subplots()
#ax.hist(mxPH, alpha=0.9, color='blue')
#plt.show() # 近乎對稱鐘型分佈

#ax = plt.gca() # 繪圖的多種方法
## the histogram of the data
#ax.hist(mxPH, bins=35, color='r')
#ax.set_xlabel('Values')
#ax.set_ylabel('Frequency')
#ax.set_title('Histogram of mxPH')
#plt.show()

fig = plt.figure() # 繪圖的多種方法
ax = fig.add_subplot(111) # 圖面佈局之 2,1,1 or 2,1,2 行、列、圖
ax.hist(mxPH) # high-level plotting 高階繪圖
ax.set_xlabel('mxPH Values') # low-level plotting 低階繪圖的畫龍點睛
ax.set_ylabel('Frequency') # low-level plotting
ax.set_title('Histogram of mxPH') # low-level plotting
plt.show()
#### 總結：圖面宣告與佈局、數據與高階繪圖、低階繪圖的畫龍點睛

#### 常態機率繪圖或分位數圖
import statsmodels.api as sm
from matplotlib import pyplot as plt
fig = sm.qqplot(mxPH) # 高階繪圖函數qqplot from statsmodels.api
plt.show()

#### 算術平均數單一補值
print(algae['mxPH'].describe())
mean = algae['mxPH'].mean() # pandas Series自動排除nan後計算mean，此舉與Python numpy套件做法不同！水很深 ~
algae['mxPH'].fillna(mean, inplace=True) # 以算術平均數填補唯一的遺缺值
print(algae['mxPH'].describe()) # 確認是否填補完成

#### Chla單一補值
Chla = algae['Chla'].dropna() # 須先移除NaNs後再繪圖或計算
fig, ax = plt.subplots() # 有發現111被省略了嗎！人性本懶
ax.hist(Chla, alpha=0.9, color='magenta')
ax.set_xlabel('Chla Values')
ax.set_ylabel('Frequency')
ax.set_title('Histogram of Chla')
plt.show() # 右偏不對稱分佈

#### 常態機率繪圖或分位數圖
import statsmodels.api as sm
from matplotlib import pyplot as plt
fig = sm.qqplot(Chla)
plt.show()

print(algae['Chla'].describe()) # count 188, 12 missings
median = algae['Chla'].median() # pandas Series自動排除nan後計算median，此舉與Python numpy套件做法不同！水很深 ~
algae['Chla'].fillna(median, inplace=True) # 以中位數(50%分位數)填補遺缺值
print(algae['Chla'].describe()) # 確認是否填補完成(count 198)

#### 多變量補值
alCorr = algae.corr() # 自動挑數值變數計算相關係數 r -> correlation coefficient matrix，PO4與oPO4高相關(-1 <= r <= 1)

alCorr
alCorr.shape

#### 相關係數方陣視覺化
import numpy as np
# Python好用的多變量繪圖套件
import seaborn as sns
# 仍需搭配基礎繪圖套件matplotlib
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
# 建立上三角遮罩矩陣
mask = np.zeros_like(alCorr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# 圖面與調色盤設定(https://zhuanlan.zhihu.com/p/27471537)
f, ax = plt.subplots(figsize = (9, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True) 
# 繪製相關係數方陣熱圖
sns.heatmap(alCorr, mask=mask, annot=True, cmap=cmap, ax=ax)
f.tight_layout()


# 繪製另一種形式的相關係數方陣熱圖
sns.clustermap(alCorr, annot=True, cmap=cmap)
f.tight_layout()

#### 相關係數符號方陣
#for i in range(15):
#  for j in range(15):
#    # 如果完全正相關或完全負相關的則不需要更改數字
#    if alCorr.iloc[i,j] == -1 or alCorr.iloc[i,j] == 1:break
#    elif alCorr.iloc[i,j] < 0.3:alCorr.iloc[i,j]=' '
#    elif alCorr.iloc[i,j] >= 0.3 and alCorr.iloc[i,j] < 0.6:
#      alCorr.iloc[i,j]='.'
#    elif alCorr.iloc[i,j] >= 0.6 and alCorr.iloc[i,j] < 0.8:
#      alCorr.iloc[i,j]=','
#    elif alCorr.iloc[i,j] >= 0.8 and alCorr.iloc[i,j] < 0.9:
#      alCorr.iloc[i,j]='+'
#    elif alCorr.iloc[i,j] >= 0.9 and alCorr.iloc[i,j] < 0.95:
#      alCorr.iloc[i,j]='*'
#    elif alCorr.iloc[i,j] >= 0.95 and alCorr.iloc[i,j] < 1:
#      alCorr.iloc[i,j]='B'
#    
#    # 對角線以後的值刪除使矩陣變成下三角矩陣
#    for k in range((i+1),15):alCorr.iloc[i,k]=' '
#  
#print(alCorr)

#### PO4多變量補值
# https://github.com/statsmodels/statsmodels/issues/5343
# !pip install --upgrade patsy
import statsmodels.formula.api as sm
result = sm.ols(formula="PO4 ~ oPO4", data=algae).fit() # ols: ordinary least square

type(result) # statsmodels.regression.linear_model.RegressionResultsWrapper
# 查詢statsmodels下RegressionResultsWrapper類物件下屬性及方法
# 初學方式
dir(result)

# 進化的過程
# Not callable表屬性
[name for name in dir(result) if not callable(getattr(result, name))]
# Callable表方法
[name for name in dir(result) if callable(getattr(result, name))]

# 最聰明的方式
[(name, type(getattr(result, name))) for name in dir(result)]

# 填補會用到的迴歸方程係數
print(result.params)

# statsmodels有完整的統計報表
print(result.summary())

type(result.params)

#### 運用迴歸方程填補遺缺值 
algae.at[27, 'PO4'] = result.params[0] + result.params[1]*algae.loc[27]['oPO4'] # pandas.DataFrame改值要用set_value(列編號, 行編號或名稱, 補入之值) 0.21.0 deprecated
algae.loc[27]['PO4']

result.params[0] + result.params[1]*algae.loc[27]['oPO4']

# 重新讀入資料集
algae = pd.read_csv("./algae.csv")
algae = algae.dropna(thresh=13)

# 創造多個PO4遺缺值的情境
algae.PO4[28:33]=np.nan # Warning!

# 考慮連自變數oPO4都遺缺的邊界案例(edge case)(參見3.5節)
algae.oPO4[32]=np.nan

algae_nar = algae.PO4.isnull()
print(algae[algae_nar == True][['oPO4', 'PO4']])

def fillPO4(oP):
    # 邊界案例判斷與處理
    if np.isnan(oP): return np.nan
    # 否則，運用模型物件result中迴歸係數進行補值計算
    else: return result.params[0] + result.params[1]*oP

# 邏輯值索引、隱式迴圈與自訂函數
algae.PO4[np.isnan(algae.PO4)==True] = algae.oPO4[np.isnan(algae.PO4)==True].apply(fillPO4)

algae.loc[27:32,['oPO4','PO4']]

