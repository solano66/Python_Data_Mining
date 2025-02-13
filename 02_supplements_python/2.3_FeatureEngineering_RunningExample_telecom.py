'''
Collated by Ching-Shih Tsou 鄒慶士 博士 (Ph.D.) Distinguished Prof. at the Department of Mechanical Engineering/Director at the Center of Artificial Intelligence & Data Science (機械工程系特聘教授兼人工智慧暨資料科學研究中心主任), MCUT (明志科技大學); Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), NTUB (國立臺北商業大學); the Chinese Academy of R Software (CARS) (中華R軟體學會創會理事長) and the Data Science and Business Applications Association of Taiwan (DSBA) (臺灣資料科學與商業應用協會創會理事長); Standing Supervisor at the Chinese Association of Quality Assessment and Evaluation (CAQAE) (社團法人中華品質評鑑協會常務監事); Chairman of the Committee of Big Data Quality Applications at the Chinese Society of Quality (CSQ) (社團法人中華民國品質學會大數據品質應用委員會主任委員)
Notes: This code is provided without warranty.
'''

#### 0. Data Importing 數據集匯入

import pandas as pd
tele = pd.read_csv('churn.csv')
tele.drop(["Unnamed: 0"], axis = 1, inplace = True)
tele.info() # like str() in R

type(tele) # pandas.core.frame.DataFrame

tele.values
type(tele.values) # numpy.ndarray

tele.columns
type(tele.columns) # pandas.core.indexes.base.Index

tele.index
type(tele.index) # pandas.core.indexes.range.RangeIndex

tele.columns.values
type(tele.columns.values) # numpy.ndarray

tele[['account_length']]
type(tele[['account_length']]) # pandas.core.frame.DataFrame (2D)

tele['account_length']
type(tele['account_length']) # pandas.core.series.Series (1D)

#### Data Sorting
# sort_index()
tele.columns
# tele = tele.drop(['Unnamed: 0'], axis=1)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

tele.sort_index(axis=1).head()

# sort_values()
teleByIntl = tele.sort_values(by=['total_intl_minutes'], ascending=False)

teleByIntl = tele.sort_values(by=['total_intl_minutes', 'total_intl_calls'], ascending=False)

#### Summary Statistics
tmp = tele.describe(include='all').T

tele.columns
tele.total_night_charge.median()
tele.total_night_charge.mean()
tele.total_night_charge.quantile([0, 0.25, 0.5, 0.75, 1])

tele.dtypes
tele.state.mode()
tele.state.value_counts()

tele.median(numeric_only=True) # pandas新版有更動 numeric_only預設是False
# tele.mode()

# 類別變量分散程度 - 熵
from scipy.stats import entropy
entropy(tele['area_code'].value_counts()) # 計算熵值時傳入次數分佈表，np.log2(3)=1.584962500721156

#### Feature Transformation
# Standardization
from sklearn.preprocessing import StandardScaler # Step 1
sc = StandardScaler() # Step 2
sc.fit(tele[['total_day_calls', 'total_intl_charge']]) # Step 3
# ['total_day_calls', 'total_intl_charge'] -> c('total_day_calls', 'total_intl_charge')
dir(sc)

sc.mean_
sc.scale_

call_charge_std = sc.transform(tele[['total_day_calls', 'total_intl_charge']]) # Step 4

# 懶人語法
# sc.fit(tele[['total_day_calls', 'total_intl_charge']])與call_charge_std = sc.transform(tele[['total_day_calls', 'total_intl_charge']])合起來寫

call_charge_std = sc.fit_transform(tele[['total_day_calls', 'total_intl_charge']]) # Steps 3 & 4

#### Boxplot visualization by seaborn
tele.info()
tele.area_code.value_counts()

import matplotlib.pyplot as plt
import seaborn as sns
sns.boxplot(x=tele['area_code'], y=tele['account_length'], hue=tele['voice_mail_plan'])
plt.legend(loc='lower right')

#### Pie plot visualization by pandas
tele.area_code.value_counts().plot.pie()

#### 0. Data Understanding and Missing Values Identifying 資料理解與遺缺值辨識
# tele.columns[0]

# tele = tele.drop(tele.columns[0], axis=1)

tele.head()
tele.info() # RangeIndex, Columns, dtypes, memory type
tele.shape
tele.columns.values # 21 variable names

tele.dtypes

tele.describe(include = 'all')

tele.isnull().any() # check NA by column

tele.isnull().values.any() # False, means no missing value ! Check the difference between above two !!!!

tele.isnull().sum() # No missing value for each variable 各變量均無遺缺值

#### 1. Select the training set 選擇訓練集
tele.dtypes

tele['case'].nunique()
tele['case'].unique()

# 足以取代上述兩個指令
tele.case.value_counts()

# select the training set (logical indexing) 邏輯值索引挑出訓練集
tele_train = tele.loc[tele['case']=='train'] # same as tele[tele['case']=='train']
tele_train.head()

# 注意tele['case']與tele[['case']]的區別！R語言亦有類似的情況！

tele['case'] # 沒有變數名稱
type(tele['case']) # <class 'pandas.core.series.Series'>
tele[['case']] # 有變數名稱
type(tele[['case']]) # <class 'pandas.core.frame.DataFrame'>

#### 2. Create feature matrix 建立屬性矩陣 X
teleX = tele_train.drop(['case', 'churn'], axis=1) # drop掉最前'case'和最後'churn'
teleX.head()

#### 2-1. Create class label 建立類別標籤 y
label_train = tele_train.loc[:,['churn']] 
label_train.head()

#### 3. Differentiate categorical features from numeric features 區分開類別與數值屬性
teleX.dtypes

# 笨鳥先飛
teleXnum = teleX.drop(teleX.columns[[0, 2, 3, 4]], axis=1)

# 聰明的select_dtypes(exclude=)
teleXnum = teleX.select_dtypes(exclude=['object'])

# 笨鳥先飛
teleXcat = teleX[teleX.columns[[0, 2, 3, 4]]]

# 聰明的select_dtypes(include=)
teleXcat = teleX.select_dtypes(include=['object'])
teleXnum = teleX.select_dtypes(exclude=['object'])

#### 4. Create class label vector(label encoding and one-hot encoding) 類別標籤編碼(標籤與單熱編碼)
# R語言需要嗎？何時需要？何時不需要？
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder # Encode labels with value between 0 and n_classes-1.

# label encoding (le) 懶人寫法
le_class = LabelEncoder().fit(tele['churn']) # Steps 2&3
churn_label = le_class.transform(tele['churn']) # Step 4
churn_label.shape

# one-hot encoding
ohe_class = OneHotEncoder(sparse_output=False).fit(churn_label.reshape(-1,1)) # sparse : boolean, default=True Will return sparse matrix if set True else will return an array.
#help(OneHotEncoder)
ohe_class.get_params()
#{'categorical_features': 'all',
# 'dtype': float,
# 'handle_unknown': 'error',
# 'n_values': 'auto',
# 'sparse': False}
#ohe_class.categorical_features

churn_ohe = ohe_class.transform(churn_label.reshape(-1,1)) # (5000, 2)

churn_label.reshape(-1,1).shape # (2019, 1) different to 1darray (2019,)

churn_ohe.shape # (5000, 2) 2darray
churn_ohe

#### 5. Pick out low variance feature(s) 低變異(只對量化屬性)屬性過濾
# scikit-learn (from scipy下kits for machine learning) -> sklearn (sk stands for scikit) 低變異過濾會設定變異數值的門檻threshold
import numpy as np
from sklearn.feature_selection import VarianceThreshold # Step 1
teleXnum_var = teleXnum.var().sort_values(ascending = False)
teleXnum_var.hist()
teleXnum_var.quantile(np.linspace(0, 1, 100))
sel=VarianceThreshold(threshold=0.6) # 0.16 -> 0.6, Step 2
print(sel.fit_transform(teleXnum)) # Steps 3 & 4
#help(sel)

# fit and transform on same object
sel.fit_transform(teleXnum).shape # (3333, 14), zero low variance features 未移除任何數值變數！(若門檻值為0.6，則會移除一個變量)


# Find the standard deviation and filter 直接計算各變數的標準差，再以邏輯值索引移除低變異數的變數
help(teleXnum.std)
teleXnum.std()
import numpy as np
~(teleXnum.std() < np.sqrt(0.6))
teleXnum_nzvx = teleXnum.loc[:,~(teleXnum.std() < np.sqrt(0.6))]

#### 6. Transform skewed feature(s) by Box-Cox Transformation 偏斜屬性的BC轉換 (對量化變數計算偏態係數)
# 判斷變量分佈是否偏斜的多種方式：1. 比較平均數與中位數; 2. 最大值與最小值的倍數，倍比大代表數值跨越多個量綱/級order of mgnitude; 3. 計算偏態係數; 4. 繪製直方圖、密度曲線、盒鬚圖等變量分佈視覺化圖形; 5. 檢視分位數值quantiles, quartiles, percentiles

teleXnum.skew(axis=0).sort_values(ascending=False)
teleXnum.columns[teleXnum.skew(0) > 1] # 偏態係數高於1的屬性(array(['number_vmail_messages', 'total_intl_calls', 'number_customer_service_calls'], dtype=object))

#### Box-Cox Transformation BC轉換
# 先試total_intl_calls前六筆(只接受一維陣列，自動估計lambda)
from scipy import stats
print(teleX['total_intl_calls'].head(6))
stats.boxcox(teleX['total_intl_calls'].head(6))
# Output (array[transformed_values], lambda_used_for_BC_transform)

#### 以下為講義p.269的練習
# BC轉換傳入的變數值必須為正數(不可為負值！！！)
# try except捕捉異常狀況(常用的程式撰寫技巧)
# https://stackoverflow.com/questions/8069057/try-except-inside-a-loop
bc = {} # 待會放恆正的屬性(可作BC轉換)
for col in teleXnum.columns:
  try:
    bc[col] = stats.boxcox(teleXnum[col])[0]
  except ValueError:
    print('Non-positive columns:{}'.format(col))
  else:
    continue

# 有非恆正的變量時，請加上一微小正數，或是運用其他偏斜轉換，例如：Y-J轉換

#### Kurtosis
teleXnum.kurtosis(axis=0).sort_values(ascending=False)

#### 7. Dimensionality Reduction by Principal Compoenets Analysis (PCA) 主成份分析維度縮減(空間分解decomposition)
from sklearn.decomposition import PCA # Step 1. Import necessary module & class
dr = PCA() # Step 2. Principal Components Analysis 主成份分析(透過矩陣分解decomposition)

# 分數矩陣
teleXpca = dr.fit_transform(teleXnum) # Steps 3 & 4. Modelling fitting & transformation
teleXpca.shape

dir(dr)

# 負荷矩陣
# 前十個主成份與**15**個原始變數的關係
dr.components_[:10] # [:10] can be removed.
dr.components_[:10].shape # (10 components, 15 original variables)
type(dr.components_) # numpy.ndarray
dr.components_.shape # (15, 15)

# 陡坡圖(scree plot 前段斜率絕對值大，但後段斜率降到很小！)決定取/看/用幾個主成份
dr.explained_variance_ratio_ # 15個主成份解釋的變異百分比
import matplotlib.pyplot as plt
plt.plot(range(1, 16), dr.explained_variance_ratio_, '-o')
plt.xlabel('# of components')
plt.ylabel('ratio of variance explained')

# list(range(1,59))
# range(1,59).tolist() # AttributeError: 'range' object has no attribute 'tolist'

type(teleXpca) # numpy.ndarray
# 可能可以降到**五維空間**中進行後續分析
teleXdr = teleXpca[:,:5]
teleXdr

# PCA後可在頭兩個主成份空間中視覺化各樣本
plt.scatter(x=teleXdr[:,0], y=teleXdr[:,1], c=tele_train['number_customer_service_calls'])
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')

# 不見得能看出什麼端倪！但是沒試你/妳怎麼知道？！

tele_train.number_customer_service_calls.value_counts()

#### 8. Feature Selection by Correlation Filtering (依相關係數方陣，原汁原味的挑出屬性)
import numpy as np
def find_correlation(df, thresh=0.9):
    """
    Given a numeric pd.DataFrame, this will find highly correlated features,
    and return a list of features to remove
    params:
    - df : pd.DataFrame
    - thresh : correlation threshold, will remove one of pairs of features with a correlation greater than this value
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
    drop_list = [i for j in select_nested for i in j]
    return drop_list


drop_list = find_correlation(teleXnum, 0.75) # 15 - 4 ('total_day_minutes', 'total_eve_minutes', 'total_night_minutes', 'total_intl_minutes') = 11
drop_list
len(drop_list) # 4

teleXnumCor = teleXnum.corr()


teleXnum_filtered = teleXnum.drop(drop_list, axis=1)

teleXnum_filtered.shape # (3333, 11)


#### 客戶集群 (以下請忽略)
pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',500)
tele.columns
tele.sort_values("total_intl_calls", ascending=False).head()

tele.sort_values("total_intl_calls", ascending=False).tail()

tele.describe(include='all')

features = set(tele.columns) - set(drop_list)


import seaborn as sns
sns.pairplot(tele[['total_day_charge',
 'total_eve_calls',
 'total_eve_charge',
 'total_intl_calls',
 'total_intl_charge',
 'total_night_calls']])

    
#tele1 = tele.drop(['churn'], axis=1)    
#tele1['churn'] = churn_label


sns.pairplot(tele1[['churn', 'number_customer_service_calls',
 'number_vmail_messages',
 'total_day_calls',
 'total_night_charge']])

#### 使用'total_day_charge','total_eve_calls','total_eve_charge','total_intl_calls','total_intl_charge','total_night_calls'進行集群(k=5)


# Package -> Module -> Class
from sklearn.cluster import KMeans
kmeans_model = KMeans(n_clusters=5, random_state=1)
# https://aji.tw/python%E4%BD%A0%E5%88%B0%E5%BA%95%E6%98%AF%E5%9C%A8__%E5%BA%95%E7%B7%9A__%E4%BB%80%E9%BA%BC%E5%95%A6/
#import urllib.parse
#encodedStr = 'https://aji.tw/python%E4%BD%A0%E5%88%B0%E5%BA%95%E6%98%AF%E5%9C%A8__%E5%BA%95%E7%B7%9A__%E4%BB%80%E9%BA%BC%E5%95%A6/'
#urllib.parse.unquote(encodedStr)
good_columns = tele._get_numeric_data().dropna(axis=1) # 移除非數值及含有NA(NaN)值的欄位
kmeans_model.fit(good_columns[['total_day_charge','total_eve_calls','total_eve_charge','total_intl_calls','total_intl_charge','total_night_calls']]) # 注意！改good_columns
labels = kmeans_model.labels_
print(labels)

help(np.unique)
np.unique(labels, return_counts=True)

#### 第一群的客戶的敘述統計值表
grp1 = tele.loc[labels == 0,:].describe(include='all')

grp1.to_csv('grp1.csv')

#### 第二群的客戶的敘述統計值表
grp2 = tele.loc[labels == 1,:].describe(include='all')

grp2.to_csv('grp2.csv')

#### 第三群的客戶的敘述統計值表
grp3 = tele.loc[labels == 2,:].describe(include='all')

grp3.to_csv('grp3.csv')

#### 第四群的客戶的敘述統計值表
grp4 = tele.loc[labels == 3,:].describe(include='all')

grp4.to_csv('grp4.csv')

#### 第五群的客戶的敘述統計值表
grp5 = tele.loc[labels == 4,:].describe(include='all')
dir(grp5)

grp5.to_csv('grp5.csv')


#### 線性迴歸(Linear Regression)
tele.columns
tele.corr()

import matplotlib.pyplot as plt
plt.scatter(tele.total_intl_charge, tele.total_intl_minutes)



# this is the standard import if you're using "formula notation" 
import statsmodels.formula.api as smf

# create a fitted model in one line
# 這邊以Sales為y也就是response，TV為x也就是feature
# ordinary least squares (ols)
lm = smf.ols(formula='total_intl_charge ~ total_intl_minutes', data=tele).fit()
dir(lm)

# print the coefficients
lm.params

# 法二：scikit-learn Linear Regression
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(tele.total_intl_minutes.values.reshape(-1,1), tele.total_intl_charge.values.reshape(-1,1))

# print intercept and coefficients
print(lm.intercept_)
print(lm.coef_)


#### ANOVA
#Create a boxplot
tele.columns
tele.loc[:,['total_day_charge','churn']].boxplot('total_day_charge', by='churn', figsize=(8, 5));


# ## ANOVA表格

import statsmodels.api as sm
from statsmodels.formula.api import ols
 # ordinary least squares
mod = ols('total_day_charge ~ churn',
                data=tele).fit()
                
aov_table = sm.stats.anova_lm(mod, typ=2)
print(aov_table)




