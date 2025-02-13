'''
Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the IDS Institute and ICD Lab. of NTUB (臺北商業大學資訊與決策科學研究所暨智能控制與決策研究室教授); at the ME Dept. and CAIDS of MCUT (曾任明志科技大學機械工程系特聘教授兼人工智慧暨資料科學研究中心主任); at the IM Dept. of SHU (曾任世新大學資訊管理學系副教授); at the BA Dept. of CHU (曾任中華大學企業管理學系副教授); the CSQ (中華民國品質學會AI暨大數據品質應用委員會主任委員); the DSBA (2013年創立臺灣資料科學與商業應用協會); and the CARS (2012年創立中華R軟體學會)
Notes: This code is provided without warranty.
'''

#### Normalization
# Python語言產生隨機抽樣模擬數據
import numpy as np
np.random.seed(1234)
X = np.random.rand(3,4) # Create an array of the given shape and populate it with random samples from a uniform distribution over [0, 1).

# 橫向正規化(固定和)
np.sum(X, axis=1)
X_rowsum100 = X/np.sum(X, axis=1).reshape(3, 1)*100

# 核驗結果
np.sum(X_rowsum100, axis=1)

# 直向正規化(固定和)
np.sum(X, axis=0)

X_colsum100 = X/np.sum(X, axis=0)*100

# 核驗結果
np.sum(X_colsum100, axis=0)

# 橫向正規化(固定最大值)
np.max(X, axis=1)
X_max100 = X/np.max(X, axis=1).reshape(3, 1)*100

# 核驗結果
np.max(X_max100, axis=1)

# 橫向正規化(固定向量長)
np.sqrt(np.sum(X**2, axis=1))
X_length1 = X/np.sqrt(np.sum(X**2, axis=1)).reshape(3, 1)

# 核驗結果
np.sum(X_length1**2, axis=1)

#### Cell Segmentation Case (本節2.3.1的running example)
import pandas as pd
import numpy as np
cell = pd.read_csv('segmentationOriginal.csv')

import sys
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', sys.maxsize)

#### 0. Data Architecture Understanding and Missing Values Identifying 數據架構基本理解與缺失值辨識
cell.info() # RangeIndex, Columns, dtypes, memory usage

# 查看cell DataFrame的欄位
print(cell.columns) # pandas包下的Index對象

cell.dtypes

cell_stats = cell.describe(include = "all").T # describe: descriptive statistics 摘要/敘述統計值
# cell_stats.to_excel("cell_stats.xls") # ModuleNotFoundError: No module named 'xlwt'; !conda install xlwt --y

cell.isnull().any() # check NaN/NA by column, same as cell.isnull().sum(axis=0). No missing !

cell.isnull().values.any() # False, means no missing value ! Check the difference between above two !!!!

#cell.isnull()
#type(cell.isnull()) # pandas.core.frame.DataFrame, so .index, .column, and .values three important attributes

#cell.isnull().values
#type(cell.isnull().values) # numpy.ndarray

cell.isnull().sum() # No missing !

#### 1. Select the training set 挑出訓練集
# 確認只有訓練train與測試test兩種樣本
cell['Case'].unique()
 
# 再瞭解訓練與測試樣本的次數分佈
cell.Case.value_counts() # 取單欄的句點語法
# select the training set by logical/boolean indexing 邏輯值索引
cell_train = cell.loc[cell['Case']=='Train'] # same as cell[cell['Case']=='Train'], logical indexing + broadcasting in Python or recycling in R of 'Train' + vectorization (邏輯值索引 + 短字串自動放長 + 向量元素各自比較)
# cell[[cell['Case']=='Train']] # KeyError: "None of [Index([(False, True, ...)], dtype='object')] are in the [columns]"
cell_train.head()

# 注意cell['Case']與cell[['Case']]的區別！R語言亦有類似的情況(drop = T or F)！

#### 2. Create class label vector (y) 建立類別標籤向量 (label encoding 標籤編碼 and one-hot encoding 單熱編碼，類似虛擬編碼)

# 類別標籤向量獨立為segTrainClass
segTrainClass = cell_train.Class
# 1009個類別標籤值
len(segTrainClass)

# R語言需要嗎？何時需要？何時不需要？
# Step 1
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder # Encode labels with value between 0 and n_classes-1.

# Step 2: label encoding
le_class = LabelEncoder()
dir(le_class)
# Step 3
le_class.fit(cell['Class'][cell['Case']=='Train']) # 'PS': 0, 'WS': 1

le_class.fit(segTrainClass)

dir(le_class)
le_class.classes_

# Step 4
Class_label = le_class.transform(cell['Class'][cell['Case']=='Train']) # 0: PS, 1: WS
Class_label.shape # (1009,)
pd.Series(Class_label).to_csv('Class_label.csv')

# Step 2: one-hot encoding (sklearn需先標籤編碼再單熱編碼，深度學習套件keras不用如此麻煩！2022/Aug.已經不用)
ohe_class = OneHotEncoder(sparse=False) # sparse : boolean, default=True Will return sparse matrix if set True else will return an array. sparse_output=False

# Step 3
ohe_class.fit(cell_train.Class.values.reshape(-1,1)) # 注意！需reshape成直行向量
# array(['PS', 'WS', 'PS', ..., 'PS', 'PS', 'WS'])
# array([['PS'],
#        ['WS'],
#        ['PS'],
#        ...,
#        ['PS'],
#        ['PS'],
#        ['WS']])


#help(OneHotEncoder)
ohe_class.get_params()
#{'categorical_features': 'all',
# 'dtype': float,
# 'handle_unknown': 'error',
# 'n_values': 'auto',
# 'sparse': False}
#ohe_class.categorical_features

# Step 4
Class_ohe = ohe_class.transform(cell_train.Class.values.reshape(-1,1)) # (1009, 2)

Class_label.reshape(-1,1).shape # (1009, 1) different to 1darray (1009,)

Class_ohe.shape # (1009, 2) 2darray
Class_ohe


# 再練習一下最快的方法 The fast way to do one-hot encoding
Class_ohe_pd = pd.get_dummies(cell['Class'])
print(Class_ohe_pd.head()) # 有欄位/變量名稱

# 如何做虛擬編碼(dummy coding)？
Class_dum_pd = pd.get_dummies(cell['Class'], drop_first = True) # 請留意結果只有'WS', 'PS'已被drop掉了！
print(Class_dum_pd.head()) # 有欄位/變量名稱

#### 3. Create feature matrix (X) 建立屬性矩陣
cell_data = cell_train.drop(['Cell','Class','Case'], axis = 'columns')
cell_data.head()

# alternative way to do the same thing
cell_data = cell_train.drop(cell_train.columns[0:3], 1) # axis=1
cell_data.head()


#### 4. Differentiate categorical features from numeric features 區分類別與數值屬性
# 變數名稱中有"Status" versus 沒有"Status"

# 不完整的做法
# 區分cell中的數值與類別欄位為不同表格
# 擷取cell中的類別欄位
cell_cat = cell.select_dtypes(include="object")

# 擷取cell中的數值欄位
cell_num = cell.select_dtypes(["float64", "int64"])
# cell_num = cell.select_dtypes(exclude="object")

# 將 cell_num 中的資料繪製成Histogram
cell_num.hist(bins=20, figsize = (30, 30))

# TODO: 還需要再區分出真正的數值變數
# cell_com.groupby(['Case']).hist(bins=20, normed=True, figsize = (30, 30))

# cell_com.groupby(['Class']).hist(bins=20, normed=True, figsize = (30, 30))

# 未被蒙蔽的做法
print(cell_data.columns)
type(cell_data.columns) # pandas.core.indexes.base.Index

# 法ㄧ：
dir(pd.Series.str)
pd.Series(cell_data.columns).str.contains("Status") # logical indices after making cell_data.columns as pandas.Series
#type(pd.Series(cell_data.columns).str.contains("Status")) # pandas.core.series.Series

cell_data.columns[pd.Series(cell_data.columns).str.contains("Status")] # again pandas.core.indexes.base.Index
#type(cell_data.columns[pd.Series(cell_data.columns).str.contains("Status")]) # pandas.core.indexes.base.Index

len(cell_data.columns[pd.Series(cell_data.columns).str.contains("Status")]) # 58 features with "Status"

cell_cat = cell_data[cell_data.columns[ pd.Series(cell_data.columns).str.contains("Status")]]
cell_cat.shape
cell_cat.head()

cell_num = cell_data.drop(cell_data.columns[pd.Series(cell_data.columns).str.contains("Status")],axis=1)
cell_num.head()

# 法二(最直覺！寫迴圈)：
status = [] # 空串列容器，準備存放名稱包含'Status'變數集合
for h in range(len(cell_data.columns)):
    if "Status" in list(cell_data.columns)[h]:
        status.append(list(cell_data.columns)[h])

cell_num = cell_data.drop(status, axis=1)
cell_num.head()
# cell_num.to_csv('cell_num.csv')

# not_status = [] # 名稱無'Status'變數集合
# for h in range(len(cell_data.columns)):
#     if "Status" not in list(cell_data.columns)[h]:
#         not_status.append(list(cell_data.columns)[h])

# cell_cat = cell_data.drop(not_status, axis=1)

cell_cat = cell_data.loc[:, status]
cell_cat.head()
# cell_cat.to_csv('cell_cat.csv')

# 法三： The most succinct way I think 最簡潔
cell_cat = cell_data.filter(regex='Status') # Note that this routine does not filter a dataframe on its contents. The filter is applied to the labels of the index.
cell_cat.head()

# 隱式implicit vs explicit迴圈的運用
cell_cat.apply(lambda x: x.value_counts(), axis=0) # 成批產製次數分配表(important in Big Data era)


# Separate positive predictors 挑出變量值恆正的預測變量集合
pos_indx = np.where(cell_data.apply(lambda x: np.all(x > 0)))[0]
cell_data_pos = cell_data.iloc[:, pos_indx]
cell_data_pos.head()
#help(np.all)

#### 5. Pick out low variance feature(s) 低變異/方差過濾
# scikit-learn (from scipy下kits for machine learning) -> sklearn (sk stands for scikit)
# Step 1
from sklearn.feature_selection import VarianceThreshold
# X=cell_num
# Step 2
sel = VarianceThreshold(threshold = 0.16) # 0.16
print(sel.fit_transform(cell_num))
#help(sel)

# What's the output?
# Y = [[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]]
# selector = VarianceThreshold()
# selector.fit_transform(Y) # remove the first and last attributes because of zero variance

# Step 3 & 4: fit and transform on same object
sel.fit_transform(cell_num).shape # (1009, 49), nine low variance features already removed 九個低變異變量已經被移除，到底哪九個！？
dir(sel)

sel.get_support() # 傳回58個真假值
idx = sel.get_support(indices=True) # 傳回留下來的49個變數編號
set(range(58))-set(idx) # 利用集合的差集運算，產生移除掉的9個變數編號


cell_num.columns[~sel.get_support()] # 邏輯值索引again！傳回移除掉的9個變數名稱(~ like ! in R)
cell_num.columns[list(set(range(58))-set(idx))]

import numpy as np
unique, counts = np.unique(sel.get_support(), return_counts=True)

dict(zip(unique, counts)) # {False: 9, True: 49}

# 另法可直接利用pandas DataFrame的標準差或變異數計算公式：Find the standard deviation and filter
help(cell_num.std)
cell_num.std()**2
cell_num.std() > .3

threshold = 0.3
print(cell_num.std()[cell_num.std() < threshold].index.values)
# cell_num.drop(cell_num.std()[cell_num.std() < threshold].index.values, axis=1) # 移除變異數過低的屬性, too large

#### 常問的問題
# 標準差或變異數門檻值如何決定？通常設為零，或者依各變量標準差或變異數在整個變數集的分佈情況決定，沒有標準答案。或者在domain中已有經驗(eg. 各點位的上下限值)，則可援引此標準。The last resort ~ 與後續建模方法結合，依最終預測績效決定合宜的門檻值！
cell_num.std().hist() # 組距可再調小！

# How to decide what threshold to use for removing low-variance features? (https://datascience.stackexchange.com/questions/31453/how-to-decide-what-threshold-to-use-for-removing-low-variance-features)

#### 過度分散(percentUnique > 10%)與過度集中(freqRation > 95/5=19)的變數
# percentUnique為獨一無二的類別值數量與樣本大小的比值(10%，i.e. 10/100，太高表過度分散！)
cell_cat.dtypes
percentUnique = cell_cat.AngleStatusCh1.nunique()/cell_cat.shape[0]

# freqRatio為最頻繁的類別值頻次，除以次頻繁類別值頻次的比值(95/5，i.e. 95%/5%，太高表過度集中！)
np.unique(cell_cat.AngleStatusCh1, return_counts=True)
freq = cell_cat.AngleStatusCh1.value_counts()
freqRatio = freq[0]/freq[1]

def percUnique(x):
    return x.nunique()/len(x) # 獨一無二的類別值/樣本數
    
def freqRatio(x):
    freq = x.value_counts() # 先做次數分佈表
    if (len(freq) == 1):
        return print('Bingo! Zero variance variable has been found.') # 零變異的變量
    else:
        return freq.iloc[0]/freq.iloc[1] # 最頻繁/次頻繁(https://stackoverflow.com/questions/24273130/get-first-element-of-series-without-knowing-the-index)

cell_cat.apply(percUnique, axis=0).sort_values(ascending=False) # Without any quasi-ID variable

cell_cat.apply(freqRatio, axis=0).sort_values(ascending=False) # With two near-zero variance variables found

cell_cat.apply(lambda x: x.value_counts(), axis=0) # MemberAvgAvgIntenStatusCh2 and MemberAvgTotalIntenStatusCh2

freqRatio(cell_cat.MemberAvgAvgIntenStatusCh2) # Bingo! Zero variance variable has been found.
cell_cat.MemberAvgAvgIntenStatusCh2.value_counts()
len(cell_cat.MemberAvgAvgIntenStatusCh2.value_counts()) # 1

freqRatio(cell_cat.MemberAvgTotalIntenStatusCh2) # Bingo! Zero variance variable has been found.
cell_cat.MemberAvgTotalIntenStatusCh2.value_counts()

#### 6. Transform skewed feature(s) by Box-Cox Transformation 偏斜分佈屬性Box-Cox 轉換
# 判斷變量分佈是否偏斜的多種方式：1. 比較平均數與中位數; 2. 最大值與最小值的倍數，倍比大代表數值跨越多個量綱/級order of mgnitude; *3. 計算偏態係數(三級動差/矩); 4. 繪製直方圖、密度曲線、盒鬚圖等變量分佈視覺化圖形; 5. 檢視分位數值quantiles, percentiles, quartiles
cell_num['VarIntenCh3'].describe() # 沒有偏態係數，只提供*平均值*、標準差及其他*位置量數(含中位數)*
cell_num['VarIntenCh3'].max()/cell_num['VarIntenCh3'].min()
cell_num['VarIntenCh3'].skew() # 理論值域：-Inf ~ Inf, 可能的合理範圍：-1 ~ 1, -2 ~ 2, -3 ~ 3 (比較誇張)
cell_num['VarIntenCh3'].hist()

# seaborn套件的displot是直方圖搭配密度曲線
import seaborn as sns
sns.distplot(cell_num.VarIntenCh3) # DISTribution Plot: 直方圖加上密度曲線來看分佈

# pandas的位置量數，結合numpy的linspace方法產生
import numpy as np
dir(cell_num.VarIntenCh3)
cell_num.VarIntenCh3.quantile(np.linspace(0, 1, 100))
import decimal
cell_num.VarIntenCh3.quantile(np.linspace(0, 1, 100, dtype=decimal.Decimal)) # 與R不完全相同！

# 最客觀的方式還是偏斜係數，所有58量化變數的偏斜係數產生一張表，降冪排列
cell_num.skew(axis=0).sort_values(ascending=False)
cell_num.skew(axis=0).sort_values(ascending=False).head()
cell_num.skew(axis=0).sort_values(ascending=False).tail()

cell_num.skew(axis=0)[cell_num.skew(axis=0) > 3].sort_values(ascending=False)
cell_num.skew(axis=0)[cell_num.skew(axis=0) > 3].index.values # 偏態係數(絕對值)高於3的屬性

# python plot multiple histograms (https://stackoverflow.com/questions/47467077/python-plot-multiple-histograms)
# 取出右偏前九高的變數名稱
highlyRightSkewed = cell_num.skew(axis=0).sort_values(ascending=False).head(n=9).index.values

import matplotlib.pyplot as plt

# Generically define how many plots along and across
ncols = 3
nrows = int(np.ceil(len(highlyRightSkewed) / (1.0*ncols)))
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))

# Lazy counter so we can remove unwated axes
counter = 0
for i in range(nrows):
    for j in range(ncols):

        ax = axes[i][j]

        # Plot when we have data
        if counter < len(highlyRightSkewed):

            ax.hist(cell_num[highlyRightSkewed[counter]], label='{}'.format(highlyRightSkewed[counter])) # , bins=10, color='blue', alpha=0.5
            ax.set_xlabel(highlyRightSkewed[counter])
            ax.set_ylabel('Frequency')
            # ax.set_ylim([0, 5])
            leg = ax.legend(loc='upper right')
            # leg.draw_frame(False) # legend draw frame or not

        # Remove axis when we no longer have data
        # else:
        #     ax.set_axis_off()

        counter += 1

plt.show()


# 取出左偏前九高的變數名稱
highlyLeftSkewed = cell_num.skew(axis=0).sort_values(ascending=False).tail(n=9).index.values

# Generically define how many plots along and across
ncols = 3
nrows = int(np.ceil(len(highlyLeftSkewed) / (1.0*ncols)))
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))

# Lazy counter so we can remove unwated axes
counter = 0
for i in range(nrows):
    for j in range(ncols):

        ax = axes[i][j]

        # Plot when we have data
        if counter < len(highlyLeftSkewed):

            ax.hist(cell_num[highlyLeftSkewed[counter]], label='{}'.format(highlyLeftSkewed[counter])) # , bins=10, color='blue', alpha=0.5
            ax.set_xlabel(highlyLeftSkewed[counter])
            ax.set_ylabel('Frequency')
            # ax.set_ylim([0, 5])
            leg = ax.legend(loc='upper left')
            leg.draw_frame(False)

        # Remove axis when we no longer have data
        else:
            ax.set_axis_off()

        counter += 1

plt.show()

# A faster way to plot several histograms in Python 快速繪圖法
# How to change the space between histograms in pandas(https://stackoverflow.com/questions/52359595/how-to-change-the-space-between-histograms-in-pandas/52359774)
cell_num[highlyRightSkewed].hist(figsize = (30, 30))
# plt.tight_layout()
# plt.show()

import numpy as np
from scipy import stats
#算skewness
#skewValues = stats.skew(cell_num)
print(stats.skew(cell_num)) # numpy.ndarray
type(stats.skew(cell_num))

skewValues = cell_num.apply(stats.skew, axis=0) # pandas.Series
print(skewValues)

#### Box-Cox Transformation
# 先試AreaCh1前六筆(只接受一維陣列，自動估計lambda)
from scipy import stats
print(cell['AreaCh1'].head(6))
stats.boxcox(cell['AreaCh1'].head(6))

# stats.boxcox()輸出為兩元素，BC轉換後的numpy ndarray AreaCh1與lambda估計值，形成的值組
type(stats.boxcox(cell['AreaCh1'].head(6))) # tuple

# 分別取出BC轉換後的AreaCh1與lambda估計值
stats.boxcox(cell_num['AreaCh1'])[0]
stats.boxcox(cell_num['AreaCh1'])[1]
help(stats.boxcox)

# 補充：另一種Box-Cox公式(可輸入二維數據，要給lambda)
from scipy.special import boxcox1p

y = cell_num.iloc[:,1]
print(y.shape)

lambda_range = np.linspace(-2, 5)  # default num=50
llf = np.zeros(lambda_range.shape, dtype=float)

# lambda estimate:
for i, lam in enumerate(lambda_range):
    llf[i] = stats.boxcox_llf(lam, y)

# find the max log-likelihood(llf) index and decide the lambda
lambda_best = lambda_range[llf.argmax()]
print('Suitable lambda is: ',round(lambda_best,2))
print('Max llf is: ', round(llf.max(),2))

# boxcox convert:
print('before convert:','\n', y.head())
#y_boxcox = stats.boxcox(y, lambda_best)
y_boxcox = boxcox1p(y, lambda_best)
print('after convert: ','\n', y_boxcox.head())

# inverse boxcox convert:
from scipy.special import inv_boxcox1p
y_invboxcox = inv_boxcox1p(y_boxcox, lambda_best)
print('after inverse: ', '\n', y_invboxcox.head())

# 若不擷取最佳的，只有拿估計好的lamda array，即用 llf array跑
# 注意！此處lambda並非最佳值
cell_num_bc_test = boxcox1p(cell_num, np.linspace(-2, 5, num=58))
# 各欄位依對應的lambda值完成轉換
cell_num_bc_test.head()
cell_num_bc_test.shape


#### 以下為講義p.257的練習
# BC轉換傳入的變數值**必須為正數**
# try except捕捉異常狀況(常用的程式撰寫技巧)
# https://stackoverflow.com/questions/8069057/try-except-inside-a-loop
bc = {}
for col in cell_num.columns:
  try:
    bc[col] = stats.boxcox(cell_num[col])[0]
  except ValueError:
    bc[col] = cell_num[col]
    print('Non-positive columns非正值變數:{}'.format(col))
  else:
    continue

cell_num_bc = pd.DataFrame(bc)
print(cell_num_bc)

import matplotlib.pyplot as plt
plt.figure()
plt.subplot(211) # nrow, ncol, nplot
plt.hist(cell_num['AreaCh1'])
plt.subplot(212)
plt.hist(cell_num_bc['AreaCh1'])
plt.show()

cell_num['AreaCh1'].describe() # mean > median
cell_num_bc['AreaCh1'].describe() # mean ~= median

#### 7. Dimensionality Reduction (dr) by PCA 主成份分析維度縮減
# Step 1
from sklearn.decomposition import PCA # The input data is centered but not scaled for each feature before applying the SVD.
# Step 2
dr = PCA() # Principal Components Analysis 主成份分析，透過矩陣分解decomposition，預設會提取出min(n_samples, n_features)=58主成份，可改成PCA(n_components = 20)

# Steps 3&4
# 分數矩陣cell_pca (cell_num 舊空間 -轉軸-> cell_pca 新空間)
cell_pca = dr.fit_transform(cell_num) # PCA只能針對量化變數計算
cell_pca

# 確認主成份之間是否獨立無關
tmp = pd.DataFrame(cell_pca).corr()
(tmp > 0.0001).sum().sum() # 58，確實獨立無關！

(np.corrcoef(cell_pca, rowvar=False) > 0.0001).sum() # 58

# 檢視模型擬合完後，有無新增的屬性與方法(通常一定有！)
dir(dr)

# 負荷矩陣或旋轉矩陣
# 前十個主成份與58個原始變數的(線性組合)關係
dr.components_[:10] # [:10] can be removed.
type(dr.components_) # numpy.ndarray
dr.components_.shape # (58主成份, 58原始變量)的方陣

# 陡坡圖(scree plot)決定取幾個主成份
dr.explained_variance_ratio_ # 各個主成份詮釋資料集總變異量的百分比(依序遞減排列)
import matplotlib.pyplot as plt
plt.plot(range(1, 26), dr.explained_variance_ratio_[:25], '-o')
plt.xlabel('# of components')
plt.ylabel('ratio of variance explained')

# 計算累積變異百分比
cumevr=[]
a=0
for i in range(0,len(dr.explained_variance_ratio_)):
  if (i==0):
    a=dr.explained_variance_ratio_[0]
  else:
    a=a+dr.explained_variance_ratio_[i]
  cumevr.append(a)
cumevr

# 快速計算方法
np.cumsum(dr.explained_variance_ratio_)

# 可能可以降到**五維**空間中進行後續分析(Why 5D? from above scree plot 上面陡坡圖)
cell_dr = cell_pca[:,:5]
cell_dr # 後續建模可考慮此PCA降維後的數據矩陣
# pd.DataFrame(cell_dr).to_csv('cell_dr.csv')

#### 8. Feature Selection by Correlation Filtering 高相關過濾
# 傳入資料矩陣繪圖，錯誤示範！
# import seaborn as sns
# sns.heatmap(cell_num)

# 正確結果
corr_matrix = cell_num.corr()
print(corr_matrix)
print(type(corr_matrix)) # class 'pandas.core.frame.DataFrame'

import seaborn as sns
# 仍需搭配基礎繪圖套件matplotlib
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
# 建立上三角遮罩矩陣
mask = np.zeros_like(corr_matrix, dtype=np.bool_) # (58, 58)全為假False值的矩陣
mask[np.triu_indices_from(mask)] = True # 上三角(triu: upper triangle)遮蓋(mask)起來
# 圖面與調色盤設定(https://zhuanlan.zhihu.com/p/27471537)
f, ax = plt.subplots(figsize = (36, 36))
cmap = sns.diverging_palette(220, 10, as_cmap=True) 
# 繪製相關係數方陣熱圖
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap=cmap, ax=ax)
f.tight_layout()


# 繪製另一種形式的相關係數方陣熱圖
# ax = sns.clustermap(corr_matrix, annot=True, cmap=cmap)
# f.tight_layout()

# 簡陋熱圖
# sns.axes_style("white")
# sns.heatmap(corr_matrix, mask=mask, square=True)

def find_correlation(df, thresh=0.9): # df: 量化變數資料矩陣
    """
    Given a numeric pd.DataFrame, this will find highly correlated features,
    and return a list of features to be removed
    params:
    - df : pd.DataFrame
    - thresh : correlation coefficients threshold, will remove one of pairs of features with a correlation greater than this value
    - select_flat: a list of features to be removed
    """

    corrMatrix = df.corr()
    corrMatrix.loc[:,:] =  np.tril(corrMatrix, k=-1) # 取下三角矩陣

    already_in = set() # 集合結構避免重複計入相同元素
    result = []

    for col in corrMatrix:
        perfect_corr = corrMatrix[col][corrMatrix[col] > thresh].index.tolist() # Index物件轉為list
        if perfect_corr and col not in already_in:
            already_in.update(set(perfect_corr)) # 更新集合
            perfect_corr.append(col)
            result.append(perfect_corr)

    select_nested = [f[1:] for f in result]
    select_flat = [i for j in select_nested for i in j]
    return select_flat


drop_list = find_correlation(cell_num, thresh=0.75) # 58 - 32 = 26
drop_list
len(drop_list) # 32

cell_num_filtered = cell_num.drop(drop_list, axis=1) # 後續建模可考慮此**原汁原味變數挑選**降維後的數據矩陣

np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
np.tril([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], -1)
np.tril([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], 1)
np.tril([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], 0)







