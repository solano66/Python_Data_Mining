'''
Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the IDS and CICD (資訊與決策科學研究所暨智能控制與決策研究室), Director of the Center for Institutional and Sustainable Development (校務永續發展中心主任), NTUB (國立臺北商業大學)
Notes: This code is provided without warranty.
'''

#### 1. Data Importing 數據集匯入

import pandas as pd
tele = pd.read_csv('churn.csv')

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

### 2. Create feature matrix 建立屬性矩陣 X
teleX = tele.drop(['Unnamed: 0', 'case', 'churn'], axis=1) # drop掉最前'case'和最後'churn'
teleX.head()

### 3. Create class label 建立類別標籤 y
label = tele.loc[:,['churn']] 
label.head()

### 4. Differentiate categorical features from numeric features 區分開類別與數值屬性
teleX.dtypes

# 聰明的select_dtypes(exclude=)
teleXnum = teleX.select_dtypes(exclude=['object'])

# 聰明的select_dtypes(include=)
teleXcat = teleX.select_dtypes(include=['object'])

import sys
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', sys.maxsize)

### 5. Label encoding 類別變量標籤編碼(也是建模！)
# R語言需要嗎？何時需要？何時不需要？
# Step 1. 載入類別
from sklearn.preprocessing import LabelEncoder # Encode labels with value between 0 and n_classes-1.
# Step 2. 定義空模規格(大多照預設設定)
# Step 3. 傳入訓練資料配適實模估計或學習模型參數
# label encoding (le) 懶人寫法
le_class = LabelEncoder().fit(label.values.ravel())
# Step 4. 以實模進行預測或轉換
churn_label = le_class.transform(label.values.ravel())
churn_label.shape

# le_cat = LabelEncoder().fit(teleXcat)

teleXcat = teleXcat.apply(LabelEncoder().fit_transform)

#### 6. Selection for categorical features (4 categorical features) 類別屬性挑選

from sklearn.feature_selection import SelectKBest, chi2

# SelectKBest(criteria, k)
fs = SelectKBest(chi2, k=2)
fs.fit(teleXcat, churn_label)

dir(fs)
fs.get_support() # array([False, False,  True,  True]), it means we select the last 2 variables.

cols_cat1 = teleXcat.columns[fs.get_support()]
fs.get_support().sum()

fs.pvalues_
pd.Series(fs.pvalues_).sort_values() # Why two NaNs?
cols_cat2 = teleXcat.columns[pd.Series(fs.pvalues_).sort_values()[:2].index]

# Verify if they are the same ?
set(cols_cat1) == set(cols_cat2) # True

teleXcat_new = fs.transform(teleXcat)
teleXcat_new.shape # (5000, 2)


#### 7. Selection for numeric features (15 numeric features) 數值屬性挑選
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

# Estimate mutual information (like correlation coefficient) for a continuous target variable.
mi = mutual_info_regression(teleXnum, churn_label) # functional programming style in Python
mi = pd.Series(mi) # (15,)
mi.index = teleXnum.columns # Change index to variable names
mi.sort_values(ascending=False)
mi.sort_values(ascending=False).plot.bar(figsize=(10, 4))

mi.sort_values().plot.bar(figsize=(10, 4))

# Estimate mutual information for a discrete target variable. (似乎這比較適合！因為class_label是離散二元值)
mi2 = mutual_info_classif(teleXnum, churn_label)
mi2 = pd.Series(mi2) # (15,)
mi2.index = teleXnum.columns # Change index to variable names
mi2.sort_values(ascending=False)
mi2.sort_values(ascending=False).plot.bar(figsize=(10, 4))


