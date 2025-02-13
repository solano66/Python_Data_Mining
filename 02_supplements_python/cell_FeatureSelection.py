'''
Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the IDS and CICD of NTUB (國立臺北商業大學資訊與決策科學研究所暨智能控制與決策研究室教授); at the ME Dept. and CAIDS of MCUT (2020~2022借調至明志科技大學機械工程系任特聘教授兼人工智慧暨資料科學研究中心主任); the CSQ (2019年起任中華民國品質學會大數據品質應用委員會主任委員); the DSBA (2013年創立臺灣資料科學與商業應用協會); and the CARS (2012年創立中華R軟體學會)
Notes: This code is provided without warranty.
Datasets: cell_cat.csv, cell_num.csv, Class_label.csv
'''

#### Feature selection for cell segmentation data
# Features importing
from sklearn.feature_selection import chi2, SelectKBest # chi2 is the selection criterion, SelectKBest is the selection method/function (Step 1)

import pandas as pd
import numpy as np
cell_cat = pd.read_csv('cell_cat.csv')
cell_cat = cell_cat.drop(cell_cat.columns[0], axis=1)

cell_num = pd.read_csv('cell_num.csv')
cell_num = cell_num.drop(cell_num.columns[0], axis=1)

#### Notice that we have a binary outcome
class_label = pd.read_csv('Class_label.csv', header=None) # header=None
class_label = class_label.drop(class_label.columns[0], axis=1)

import sys
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', sys.maxsize)

#### Selection for categorical features (58 categorical features) 58個類別特徵挑選
# SelectKBest(criteria, k)
fs = SelectKBest(chi2, k=20) # 定義(空)模型 Step 2 (fs: feature selection)
fs.fit(cell_cat, class_label) # 傳入訓練樣本，配適(實)模型 Step 3

fs.get_support() # array([False, False,  True,  True, ....]), it means we select the following 20 variables.
fs.get_support().sum() # 20 Trues, means that 20 variables have been selected. 挑出20個變量

cols1 = cell_cat.columns[fs.get_support()] # Step 4

# 驗證依p值大小挑出變量
fs.pvalues_
pd.Series(fs.pvalues_).sort_values() # Why two NaNs? Ans. Because these two categorical features are uniary.
cols2 = cell_cat.columns[pd.Series(fs.pvalues_).sort_values()[:20].index]

# Verify if they are the same ?
set(cols1) == set(cols2) # True

cell_cat_new = fs.transform(cell_cat) # (1009, 58) -> (1009, 20)
cell_cat_new.shape # (1009, 20)

#### Selection for numeric features (58 numeric features) 58個數值特徵挑選
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif # Step 1

# Estimate mutual information 相互訊息 (like correlation coefficient) for a continuous target variable.
# mi = mutual_info_regression(cell_num, class_label) # functional programming style in Python
# mi = pd.Series(mi) # (58,)
# mi.index = cell_num.columns # Change index to variable names
# mi.sort_values(ascending=False)
# mi.sort_values(ascending=False).plot.bar(figsize=(10, 4))

# mi.sort_values().plot.bar(figsize=(10, 4))

# Estimate mutual information for a discrete target variable. (似乎這比較適合！因為class_label是離散二元值)
mi2 = mutual_info_classif(cell_num, class_label) # Steps 2&3
mi2 = pd.Series(mi2) # (58,)
mi2.index = cell_num.columns # Change index to variable names 將索引變更為變量名稱
mi2.sort_values(ascending=False) # Step 4
mi2.sort_values(ascending=False).plot.bar(figsize=(10, 4))


#### Verify the results from R language
# from sklearn.datasets import load_iris

# X, y = load_iris(return_X_y=True, as_frame=True)
# X.shape

# # Estimate mutual information for a continuous target variable.
# mi = mutual_info_regression(X, y)
# mi = pd.Series(mi)
# mi.index = X.columns
# mi.sort_values(ascending=False)
# mi.sort_values(ascending=False).plot.bar(figsize=(10, 4))

# mi

# # Estimate mutual information for a discrete target variable. (似乎這比較適合！)
# mi2 = mutual_info_classif(X, y)
# mi2 = pd.Series(mi2)
# mi2.index = X.columns
# mi2.sort_values(ascending=False)
# mi2.sort_values(ascending=False).plot.bar(figsize=(10, 4))

# mi2

# 與R語言information.gain()計算結果均不相同，不過排名大致一樣！再追兩語言的計算方式為何？

# > (weights <- information.gain(Species~., iris))
#              attr_importance
# Sepal.Length       0.4521286
# Sepal.Width        0.2672750
# Petal.Length       0.9402853
# Petal.Width        0.9554360

#### Feature selection in Python using the Filter method (https://towardsdatascience.com/feature-selection-in-python-using-filter-method-7ae5cbc4ee05)
