'''
Collated by Ching-Shih Tsou 鄒慶士 博士 (Ph.D.) Distinguished Prof. at the Department of Mechanical Engineering/Director at the Center of Artificial Intelligence & Data Science (機械工程系特聘教授兼人工智慧暨資料科學研究中心主任), MCUT (明志科技大學); Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), NTUB (國立臺北商業大學)
Notes: This code is provided without warranty.
'''

#### Python scikit-learn 屬性挑選
from sklearn.datasets import load_iris
# Step 1: 載入所需的套件與類別函數
from sklearn.feature_selection import SelectKBest # Part 1: 選用屬性排名法，挑前K佳的

# Compute chi-square test is going to be our criteria to select the K best features
# Compute chi-squared stats between each non-negative feature and class.
# This score can be used to select the n_features features with the highest values for the test chi-squared statistic from X, which must contain only non-negative features such as booleans or frequencies (似乎不太適合iris資料集的長寬特徵) (e.g., term counts in document classification), relative to the classes.
from sklearn.feature_selection import chi2 # Part 2: 選用挑選準則 - chi2 卡方統計量

help(load_iris)

import pandas as pd
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)

y = pd.Series(data.target)

#https://stackoverflow.com/questions/62322882/load-iris-got-an-unexpected-keyword-argument-as-frame
#X, y = load_iris(return_X_y=True, as_frame=True)
#X.shape
#X.columns # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

# Step 2: 宣告空模
# SelectKBest(criteria, k)
fs = SelectKBest(chi2, k=2)
# Step 3: 配適實模
fs.fit(X, y) 

dir(fs) # 做出來的東西都藏在實模fs裡
fs.get_support() # array([False, False,  True,  True]), it means we select 'petal length (cm)' and 'petal width (cm)'

# Step 4: 預測或轉換
X_new = fs.transform(X) # Python資料導向程式設計輸入輸出不同調的常見現象(X is a pandas DataFrame, X_new is a numpy ndarray.)
X_new.shape

# 將numpy ndarray轉成pandas DataFrame(可給變數名稱，從哪來？)
import pandas as pd
Ｘ_new = pd.DataFrame(X_new, columns = X.columns[fs.get_support()])

#### 從錯誤中學習成長！
# For instance, we can use a F-test to retrieve the two best features for a dataset as follows:

# from sklearn.datasets import load_iris
# from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
# X, y = load_iris(return_X_y=True)
# X.shape

X_new = SelectKBest(f_classif, k=2).fit_transform(X, y)
X_new.shape # 結果雖然相同，但勿以惡小而為之！

# These objects take as input a scoring function that returns univariate scores and p-values (or only scores for SelectKBest and SelectPercentile):

# 1. For regression: r_regression, f_regression (X and y both are numeric), mutual_info_regression
# 2. For classification: chi2, f_classif (X is numeric and y is categorical), mutual_info_classif

#### Reference:
# https://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection

