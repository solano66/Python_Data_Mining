'''
Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the IDS, CICD, and CISD of NTUB (臺北商業大學資訊與決策科學研究所暨智能控制與決策研究室教授兼校務永續發展中心主任); at the ME Dept. and CAIDS of MCUT (2020年借調至明志科技大學機械工程系擔任特聘教授兼人工智慧暨資料科學研究中心主任兩年); the CSQ (2019年起任中華民國品質學會大數據品質應用委員會主任委員); the DSBA (2013年創立臺灣資料科學與商業應用協會); and the CARS (2012年創立中華R軟體學會)
Notes: This code is provided without warranty.
Dataset: segmentationOriginal.csv
'''

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
segTrainClass[:5]

# 標籤編碼
Class_label = segTrainClass.map({'PS': 1, 'WS': 2}).astype(int)

# 單熱編碼最快的方法 The fast way to do one-hot encoding
Class_ohe_pd = pd.get_dummies(cell['Class'])
print(Class_ohe_pd.head()) # 有欄位/變量名稱

# 如何做虛擬編碼(dummy coding)？
Class_dum_pd = pd.get_dummies(cell['Class'], drop_first = True) # 請留意結果只有'WS', 'PS'已被drop掉了！
print(Class_dum_pd.head()) # 有欄位/變量名稱

# R語言需要嗎？何時需要？何時不需要？

#### 3. Create feature matrix (X) 建立屬性矩陣
cell_data = cell_train.drop(['Cell','Class','Case'], axis = 'columns')
cell_data.head()

#### 4. Differentiate categorical features from numeric features 區分離散類別與量化數值屬性
# 變數名稱中有"Status" versus 沒有"Status"

# 法ㄧ：寫迴圈，最直覺！
status = [] # 名稱包含'Status'變數集合
for h in range(len(cell_data.columns)):
    if "Status" in list(cell_data.columns)[h]:
        status.append(list(cell_data.columns)[h])

cell_num = cell_data.drop(status, axis=1)
cell_num.head()
# cell_num.to_csv('cell_num.csv')

not_status = [] # 名稱無'Status'變數集合
for h in range(len(cell_data.columns)):
    if "Status" not in list(cell_data.columns)[h]:
        not_status.append(list(cell_data.columns)[h])

cell_cat = cell_data.drop(not_status, axis=1)
cell_cat.head()
# cell_cat.to_csv('cell_cat.csv')

# 法二： The most succinct way I think 最簡潔
cell_cat = cell_data.filter(regex='Status') # Note that this routine does not filter a dataframe on its contents. The filter is applied to the labels of the index.
cell_cat.head()

# 確認是否均為類別變量
# 隱式implicit vs explicit迴圈的運用
cell_cat.apply(lambda x: x.value_counts(), axis=0) # 成批產製次數分配表(important in Big Data era)

#### 5. Pick out low variance feature(s) 低變異/方差過濾
# scikit-learn (from scipy下kits for machine learning) -> sklearn (sk stands for scikit)

# Step 1 套件載入
from sklearn.feature_selection import VarianceThreshold

# Step 2 宣告空模
sel = VarianceThreshold(threshold = 0.16) # 0.16

# Step 3 & 4: 傳入樣本擬合實模 & 轉換 fit and transform on same object
sel.fit_transform(cell_num).shape # (1009, 49), nine low variance features already removed 九個低變異變量已經被移除，到底哪九個！？
dir(sel)

sel.get_support() # 傳回58個真假值
import numpy as np
unique, counts = np.unique(sel.get_support(), return_counts=True)
dict(zip(unique, counts)) # {False: 9, True: 49}

idx = sel.get_support(indices=True) # 傳回留下來的49個變數編號
set(range(58))-set(idx) # 利用集合的差集運算，產生移除掉的9個變數編號

cell_num.columns[~sel.get_support()] # 邏輯值索引again！傳回移除掉的9個變數名稱(~ like ! in R)
cell_num.columns[list(set(range(58))-set(idx))]

#### 常問的問題
# 標準差或變異數門檻值如何決定？依各變量標準差在整個變數集的分佈情況決定，沒有標準答案。或者在domain中已有經驗(eg. 各點位的上下限值)，則可援引此標準。The last resort ~ 與後續建模方法結合，依最終預測績效決定合宜的門檻值！
cell_num.std().hist()

# How to decide what threshold to use for removing low-variance features? (https://datascience.stackexchange.com/questions/31453/how-to-decide-what-threshold-to-use-for-removing-low-variance-features)

#### 過度分散(percentUnique > 10%)與過度集中(freqRation > 95/5=19)的變數
# percentUnique為獨一無二的類別值數量與樣本大小的比值(10%，太高表過度分散！)
cell_cat.dtypes
percentUnique = cell_cat.AngleStatusCh1.nunique()/cell_cat.shape[0]

# freqRatio為最頻繁的類別值頻次，除以次頻繁類別值頻次的比值(95/5，太高表過度集中！)
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
# 判斷變量分佈是否偏斜的多種方式：1. 比較平均數與中位數; 2. 最大值與最小值的倍數，倍比大代表數值跨越多個量綱/級order of mgnitude; 3. 計算偏態係數(三階動差/矩moments); 4. 繪製直方圖、密度曲線、盒鬚圖等變量分佈視覺化圖形; 5. 檢視分位數值quantiles, percentiles, quartiles
cell_num['VarIntenCh3'].describe() # 沒有偏態係數，只提供*平均值*、標準差及其他*位置量數(含中位數)*
cell_num['VarIntenCh3'].max()/cell_num['VarIntenCh3'].min()
cell_num['VarIntenCh3'].skew() # 理論值域：-Inf ~ Inf, 可能的合理範圍：-1 ~ 1, -2 ~ 2, -3 ~ 3 (比較誇張)
cell_num['VarIntenCh3'].hist()

# seaborn套件的displot是直方圖搭配密度曲線
import seaborn as sns
sns.distplot(cell_num.VarIntenCh3) # DISTribution Plot: 直方圖加上密度曲線來看分佈

# 最客觀的方式還是偏斜係數，所有58量化變數的偏斜係數產生一張表，降冪排列
cell_num.skew(axis=0).sort_values(ascending=False)

# python plot multiple histograms (https://stackoverflow.com/questions/47467077/python-plot-multiple-histograms)
# 取出右偏前九高的變數名稱
highlyRightSkewed = cell_num.skew(axis=0).sort_values(ascending=False).head(n=9).index.values

import matplotlib.pyplot as plt
cell_num[highlyRightSkewed].hist(figsize = (30, 30))
plt.tight_layout()
plt.show()

# 取出左偏前九高的變數名稱
highlyLeftSkewed = cell_num.skew(axis=0).sort_values(ascending=False).tail(n=9).index.values

cell_num[highlyLeftSkewed].hist(figsize = (30, 30))
plt.tight_layout()
plt.show()

#### Box-Cox Transformation
# 先試AreaCh1前六筆(只接受一維陣列，自動估計lambda)
from scipy import stats
print(cell['AreaCh1'].head(6))
stats.boxcox(cell['AreaCh1'].head(6))

# Separate positive predictors 挑出變量值恆正的預測變量集合
pos_indx = np.where(cell_data.apply(lambda x: np.all(x > 0)))[0]
cell_data_pos = cell_data.iloc[:, pos_indx]
cell_data_pos.head()
#help(np.all)

#### 7. Dimensionality Reduction (dr) by PCA 主成份分析維度縮減
# Step 1
from sklearn.decomposition import PCA

# Step 2
dr = PCA() # Principal Components Analysis 主成份分析，透過矩陣分解decomposition，預設會提取出min(n_samples, n_features)=58主成份，可改成PCA(n_components = 20)

# Steps 3&4
# 分數矩陣cell_pca (cell_num 舊空間 --轉軸--> cell_pca 新空間)
cell_pca = dr.fit_transform(cell_num) # PCA只能針對量化變數計算
cell_pca

# 確認主成份之間是否獨立無關
cor = pd.DataFrame(cell_pca).corr()
(cor > 0.0001).sum().sum() # 58，確實獨立無關！

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

# 可能可以降到**五維**空間中進行後續分析(Why 5D? from above scree plot 上面陡坡圖)
cell_dr = cell_pca[:,:5]
cell_dr # 後續建模可考慮此PCA降維後的數據矩陣
# pd.DataFrame(cell_dr).to_csv('cell_dr.csv')

#### 8. Feature Selection by Correlation Filtering 高相關過濾
# 正確結果
corr_matrix = cell_num.corr() # pandas DataFrame下的相關係數計算方法

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



