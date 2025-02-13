'''
Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the IDS and CICD of NTUB (國立臺北商業大學資訊與決策科學研究所暨智能控制與決策研究室教授); at the ME Dept. and CAIDS of MCUT (2020~2022借調至明志科技大學機械工程系任特聘教授兼人工智慧暨資料科學研究中心主任); the CSQ (2019年起任中華民國品質學會大數據品質應用委員會主任委員); the DSBA (2013年創立臺灣資料科學與商業應用協會); and the CARS (2012年創立中華R軟體學會) 
Notes: This code is provided without warranty.
'''

#### 3.3.1 重抽樣與資料切分方法
#### p.333 前半段
import pandas as pd
# import numpy as np
from resample import bootstrap # !pip install resample==0.21 or !conda install -c conda-forge resample --y (舊版Python用前者，新版用後者) 0.21 (o) -> 1.0.1 (x)
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

airquality = pd.read_csv('airquality.csv', encoding='utf-8')
# 瞭解airquality 的資料結構
airquality.info()

# 分別視覺化Ozon 與Wind 和Temp 的關係
plt.subplot(121) # 1: 一橫，2: 兩縱，1: 現在畫第一張圖
plt.scatter(airquality["Wind"],airquality["Ozone"])
plt.xlabel('Wind')
plt.ylabel('Ozone')

plt.subplot(122) # 1: 一橫，2: 兩縱，2: 現在畫第二張圖
plt.scatter(airquality["Temp"],airquality["Ozone"])
plt.xlabel('Temp')
# plt.close()

# 以各變項中位數填補遺缺值
cleanAQ = pd.DataFrame()
for col in airquality.columns:
    cleanAQ[col] = airquality[col].fillna(airquality[col].median())

# 確認資料表中已無遺缺值
cleanAQ.isnull().sum()

# https://github.com/dsaxton/resample/blob/master/examples/resample.ipynb
# 定義拔靴抽樣函數bootstrap() 所用的統計值計算函數fitreg()(類似R語言中的rsq())
def fitreg(A): # A表拔靴抽樣的樣本子集
    X = A[:,2:4] # select Wind and Temp 選擇 Wind 和Temp(注意前包後不包！)
    y = A[:,0] # select Ozon 選擇
    lm = LinearRegression()
    lm.fit(X, y)
    r_squared = lm.score(X, y) # Return the coefficient of determination of the prediction
    return {"rsq": r_squared}

# cleanAQ - bootstrapping -> 1000 resamples (sample size ? Ans. 153 same as original sample size) -> Fit LR models -> 1000 R^2 -> central tendency -> dispersion (variance or standard deviation)
# boot_coef = bootstrap.bootstrap(a=cleanAQ.values, f=fitreg, b=1000) # array of dict (從a進行b次拔靴抽樣，每次抽出之樣本進行f中定義的配適工作)
boot_coef = bootstrap.bootstrap(fitreg, cleanAQ.values, size=1000) # array of dict (從a進行b次拔靴抽樣，每次抽出之樣本進行f中定義的配適工作)
# 將boot_coef字典轉成dataframe
param_frame = pd.DataFrame([pd.Series(x) for x in boot_coef])
param_frame.head()
param_frame.shape # (1000, 1)
param_frame.hist()

# Python中如何繪製常態分位數圖(Normal Probability Plots)
import statsmodels.api as sm
from matplotlib import pyplot as plt
fig = sm.qqplot(param_frame.rsq, line="q") # Attention to line="q" and the default comparison distribution is scipy.stats.distributions.norm (a standard normal).
plt.show()

# p.341
# Sklearn中的CV與KFold詳解
# https://blog.csdn.net/FontThrone/article/details/79220127
X = pd.read_csv('predictors.csv', encoding='utf-8')
y = pd.read_csv('classes.csv', encoding='utf-8')

#### 3.3.1 後半段
# 簡單保留法
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 

# 交叉驗證法
from sklearn.model_selection import StratifiedKFold
stratified_folder = StratifiedKFold(n_splits=10, random_state=0, shuffle=True) # Try StratifiedKFold(n_splits=10, shuffle=False)

# Please double click the stratified_folder object and be attentive to the method split
# 迴圈搭配split()方法檢視十摺交叉驗證樣本編號
for train_index, test_index in stratified_folder.split(X_train, y_train): # Generate indices to split data into training and test set.
    print("Stratified Train Index:", train_index)
    print("Stratified Test Index:", test_index)
    

#### 3.3.2 單類模型參數調校 & 3.3.2.1 多個參數待調 
# p.347
# https://blog.csdn.net/luanpeng825485697/article/details/79831703
import pandas as pd
Smarket = pd.read_csv('./Smarket.csv', encoding='utf-8')

# k近鄰法不允許遺缺值
Smarket.isnull().sum()
Smarket.Year.value_counts()

# S&P Stock Market Data: Daily percentage returns for the S&P 500 stock index between 2001 and 2005. The goal is to predict whether the index will increase or decrease on a given day using the past 5 days’ percentage changes in the index.
X = Smarket.iloc[:,1:8] # 前包後不包
y = Smarket.iloc[:,8]

# 校驗集與最終測試集切分
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1234, test_size=0.25) # 校驗集與最終測試集

# 校驗集屬性矩陣與前處理(i.e. kNN因歐幾里德直線距離需標準化各個變量)
from sklearn import preprocessing
X_train = preprocessing.scale(X_train) # Python相對少見的泛函式編程語法

from sklearn.model_selection import cross_val_score, GridSearchCV # 參數grid下，以CV search最佳超參數，cross_val_score負責評分運算
from sklearn.neighbors import KNeighborsClassifier

# 最佳化參數k的調校範圍
import numpy as np
k_range = np.arange(start=5,stop=45,step=2)
# 待估參數權重的取值範圍。 uniform為統一取權值，distance表示距離倒數取權值(與R不同之處在於多調了一個參數weight_options！)
weight_options = ['uniform', 'distance'] # 各鄰居票票等值 versus 依距離的倒數來加權 
# 下面構建parameter grid，其結構是key為參數名稱，value是待搜索的數值列表的一個字典結構
# 定義最佳化參數字典，字典中的key值必須是分類算法的函數的參數名
param_grid = {'n_neighbors': k_range, 'weights': weight_options} 
print(param_grid)

# 定義分類算法。 n_neighbors和weights的參數名稱和param_grid字典中的key名對應
knn = KNeighborsClassifier() # n_neighbors=5, number of neighbors can be specified arbitrarily

# 這裡GridSearchCV的參數形式和cross_val_score的形式差不多，其中param_grid是parameter grid所對應的參數
# GridSearchCV中的n_jobs設置為-1時，可以實現並行計算（如果你的電腦支持的情況下，-1 means using all processors.）
#針對每個參數對進行了10次交叉驗證。 scoring='accuracy'使用準確率為結果的度量指標。可以添加多個度量指標
grid = GridSearchCV(estimator = knn, param_grid = param_grid, cv=10, scoring='accuracy') # grid的空模，try ‘precision’ (nans)
import time
start = time.time()
grid.fit(X_train, y_train) # grid已變成實模
end = time.time()
end - start

# 結果都藏在實模grid的後面了！
dir(grid)

print('分數紀錄：',grid.cv_results_)
print('最佳分數:',grid.best_score_)  
print('最佳參數：',grid.best_params_)
print('最佳模型：',grid.best_estimator_)  

grid.cv_results_
type(grid.cv_results_) # A dict
grid.cv_results_.keys()

# 交叉驗證結果字典，轉換成DataFrame
cv_res = pd.DataFrame(grid.cv_results_)

# 驗證交叉驗證績效評估計算結果
cv_res.iloc[0, 7:17].mean() # 0.8783916723861817 每次結果不同！
cv_res.iloc[0, 7:17].std(ddof=0) # 0.030754122995618343 (pandas Series預設變異數與標準差計算之ddof=1(分母n-1)，不過sklearn卻用ddof=0(分母n))

# 使用獲取的最佳參數，搭配全部的核驗集樣本937個，重新訓練模型，預測測試樣本標籤(圖3.15的哪個block?)
knn = KNeighborsClassifier(n_neighbors=grid.best_params_['n_neighbors'], weights=grid.best_params_['weights'])
knn.fit(X_train, y_train) # 倒進全部核驗集/訓練集樣本，配適最佳模型 knn

# 最終績效評定 model assessment (圖3.14的哪個block?)
knn.score(X_test, y_test)
test_pred = knn.predict(X_test)

#### 自行完成最終測試績效評估工作
# Homewrok due date on August/12th 17:00: y_test, test_pred 產生混淆矩陣，計算整體指標與類別指標
from sklearn.metrics import classification_report, confusion_matrix
temp = classification_report(y_test, test_pred)
type(temp) # a str easy to check results
1/((1/0.95 + 1/0.84)/2)
pd.crosstab(y_test, test_pred)

(128+154)/X_test.shape[0]

confusion_matrix(y_test, test_pred)

#### 3.3.3 比較不同類的模型(與書本/講義上的範例不同，且假設各算法預設參數可獲得最佳模型！)

import pandas
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC # Support Vector Clasiffier

# load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'] # set the names of variables
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values # 打回原形numpy ndarray
X = array[:,0:8]
Y = array[:,8]

# prepare configuration for cross validation test harness
seed = 7

# prepare models
models = [] # tuple ('簡記名', 建模類別函數全名)
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# evaluate each model in turn
results = [] # 逐次添加各模型交叉驗證績效評量結果
names = []
scoring = 'balanced_accuracy'
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True) # 十摺交叉驗證樣本切分
    # ValueError: Setting a random_state has no effect since shuffle is False. You should leave random_state to its default (None), or set shuffle=True.
	cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
    
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

for i in range(6):
    print(results[i])

# 可自行完成統計檢定：ANOVA檢定檢視是否有顯著的差異

import scipy.stats as stats

fvalue, pvalue = stats.f_oneway(results[0], results[1], results[2], results[3], results[4], results[5])

print("檢定統計量：{}，檢定p值：{}".format(fvalue, pvalue))

# 後續可以Tukey HSD找出正確率最大方法

dfn, dfd = 5, 55
stats.f.sf(fvalue,dfn,dfd)

# P(F>3.7728334504659156)=0.0052222146020425185 (p-value < 0.05) 每次結果不盡相同
# Therefore, the hypothesis H0: mean1 = mean2 = ... = mean6 is rejected.
# That is, at least one of these 6 models is better than others. 拒絕各模型無差異的虛無假說，可能至少有ㄧ個模型比其他模型表現得更好。

#### Reference: How To Compare Machine Learning Algorithms in Python with scikit-learn (https://machinelearningmastery.com/compare-machine-learning-algorithms-python-scikit-learn/)