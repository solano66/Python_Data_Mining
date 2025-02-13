### Collated by Ching-Shih Tsou 鄒慶士 博士 (Ph.D.) Distinguished Prof. at the Department of Mechanical Engineering/Director at the Center of Artificial Intelligence & Data Science (機械工程系特聘教授兼人工智慧暨資料科學研究中心主任), MCUT (明志科技大學); Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), NTUB (國立臺北商業大學)
### Notes: This code is provided without warranty.
### Datasets: Train.csv, Test.csv

import pandas as pd
train = pd.read_csv('Train.csv') # .表marketing資料夾
train.info() # 有遺缺值
train = train.dropna() # (4650, 12)
train.dtypes # 文數字雜陳

test = pd.read_csv('Test.csv') # with known 'Item_Outlet_Sales'
test.info() # 有遺缺值
test = test.dropna() # (3099, 11)
test.dtypes # 文數字雜陳

set(train.columns) - set(test.columns) # {'Item_Outlet_Sales'} 集合差集運算

#### spliting training and testing data
from sklearn.model_selection import train_test_split

X = train.iloc[:, :-1]
y = train.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=27)

X_train.dtypes
X_train.describe()

# There is a huge difference in the range of values present in our numerical features: Item_Weight, Item_Visibility, Item_MRP, and Outlet_Establishment_Year.


#### Standardization using sklearn 標準化
# data standardization with  sklearn
from sklearn.preprocessing import StandardScaler # scikit-learn六大模組: preporcessing, DR, Clustering, Reg., Classification, Model Selection

# copy of datasets
X_train_stand = X_train.copy()
X_test_stand = X_test.copy()

# numerical features 挑出量化變量
X_train.select_dtypes(exclude=['object']).columns
num_cols = X_train.select_dtypes(exclude=['object']).columns

# apply standardization on numerical features
for i in num_cols:
    
    # fit on training data column
    scale = StandardScaler().fit(X_train_stand[[i]])
    
    # transform the training data column
    X_train_stand[i] = scale.transform(X_train_stand[[i]])
    
    # transform the testing data column
    X_test_stand[i] = scale.transform(X_test_stand[[i]])

X_train_stand.describe()

#### Normalization using sklearn 正規化

# data normalization with sklearn
from sklearn.preprocessing import MinMaxScaler

# copy of datasets
X_train_norm = X_train.copy()
X_test_norm = X_test.copy()

# numerical features
num_cols = X_train.select_dtypes(exclude=['object']).columns

# apply normalization on numerical features
for i in num_cols:
    
    # fit on training data column
    scale = MinMaxScaler().fit(X_train_norm[[i]]) # Steps 2 & 3
    
    # transform the training data column
    X_train_norm[i] = scale.transform(X_train_norm[[i]]) # Step 4
    
    # transform the testing data column
    X_test_norm[i] = scale.transform(X_test_norm[[i]]) # Step 4


X_test_norm.describe()

# Because One-Hot encoded features are already in the range between 0 to 1. So, normalization would not affect their value.

#### Distribution comparisons by boxplots 盒鬚圖比較兩種尺度調整的結果
dir(num_cols) # tolist()
X_train.boxplot(num_cols.tolist()) # to_list(), wrong ! 

X_train_stand.boxplot(num_cols.tolist())

X_train_norm.boxplot(num_cols.tolist())

# The Big Question – Normalize or Standardize? 大哉問，到底要用哪一個？
# Normalization vs. standardization is an eternal question among machine learning newcomers. Let me elaborate on the answer in this section.

# Normalization is good to use when you know that the distribution of your data does not follow a Gaussian distribution. This can be useful in algorithms that do not assume any distribution of the data like K-Nearest Neighbors and Neural Networks. 無資料假設為高斯分佈時用正規化
# Standardization, on the other hand, can be helpful in cases where the data follows a Gaussian distribution. However, this does not have to be necessarily true. Also, unlike normalization, standardization does not have a bounding range. So, even if you have outliers in your data, they will not be affected by standardization. 資料假設為高斯分佈時用標準化
# However, at the end of the day, the choice of using normalization or standardization will depend on your problem and the machine learning algorithm you are using. There is no hard and fast rule to tell you when to normalize or standardize your data. You can always start by fitting your model to raw, normalized and standardized data and compare the performance for best results. 也可以兩者均作，然後比較結果再取其一

#### Applying Scaling to Machine Learning Algorithms (兩種尺度調整法 vs. k近鄰、支援向量機、決策樹)

#### training a KNN model
from sklearn.neighbors import KNeighborsRegressor
# measuring RMSE score
from sklearn.metrics import mean_squared_error

# knn 
knn = KNeighborsRegressor(n_neighbors=7)

rmse = []

# raw, normalized and standardized training and testing data
trainX = [X_train, X_train_norm, X_train_stand]
testX = [X_test, X_test_norm, X_test_stand]

import numpy as np
# model fitting and measuring RMSE
for i in range(len(trainX)):
    
    # fit
    knn.fit(trainX[i].loc[:,num_cols],y_train)
    # predict
    pred = knn.predict(testX[i].loc[:,num_cols])
    # RMSE
    rmse.append(np.sqrt(mean_squared_error(y_test,pred)))

# visualizing the result
df_knn = pd.DataFrame({'RMSE':rmse},index=['Original','Normalized','Standardized'])
df_knn # kNN: Standardization

# You can see that scaling the features has brought down the RMSE score of our KNN model. Specifically, the 'standardize' data performs a tad bit better than the normalized data.

#### training an SVR model
from  sklearn.svm import SVR
# measuring RMSE score
from sklearn.metrics import mean_squared_error

# SVR
svr = SVR(kernel='rbf',C=5)

rmse = []

# raw, normalized and standardized training and testing data
trainX = [X_train, X_train_norm, X_train_stand]
testX = [X_test, X_test_norm, X_test_stand]

# model fitting and measuring RMSE
for i in range(len(trainX)):
    
    # fit
    svr.fit(trainX[i].loc[:,num_cols],y_train)
    # predict
    pred = svr.predict(testX[i].loc[:,num_cols])
    # RMSE
    rmse.append(np.sqrt(mean_squared_error(y_test,pred)))

# visualizing the result    
df_svr = pd.DataFrame({'RMSE':rmse},index=['Original','Normalized','Standardized'])
df_svr # SVR: normalization because of nonparametric statistics

# We can see that scaling the features does bring down the RMSE score. And the 'normalized' data has performed better than the 'standardized' data. Why do you think that’s the case?

# The sklearn documentation states that SVM, with RBF kernel, assumes that all the features are centered around zero and variance is of the same order. This is because a feature with a variance greater than that of others prevents the estimator from learning from all the features. Great!

#### training a Decision Tree model
from sklearn.tree import DecisionTreeRegressor
# measuring RMSE score
from sklearn.metrics import mean_squared_error

# Decision tree
dt = DecisionTreeRegressor(max_depth=10,random_state=27)

rmse = []

# raw, normalized and standardized training and testing data
trainX = [X_train,X_train_norm,X_train_stand]
testX = [X_test,X_test_norm,X_test_stand]

# model fitting and measuring RMSE
for i in range(len(trainX)):
    
    # fit
    dt.fit(trainX[i].loc[:,num_cols],y_train)
    # predict
    pred = dt.predict(testX[i].loc[:,num_cols])
    # RMSE
    rmse.append(np.sqrt(mean_squared_error(y_test,pred)))

# visualizing the result    
df_dt = pd.DataFrame({'RMSE':rmse},index=['Original','Normalized','Standardized'])
df_dt # DTs: original, why?

# You can see that the RMSE scores are almost the same. So rest assured when you are using tree-based algorithms on your data! 樹狀模型不太需要做前處理

#### References:
# Feature Scaling for Machine Learning: Understanding the Difference Between Normalization vs. Standardization (https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/)



