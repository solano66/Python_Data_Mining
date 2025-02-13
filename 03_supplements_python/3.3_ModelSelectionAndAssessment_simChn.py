'''
Collated by Prof. Ching-Shih Tsou 邹庆士 教授 (Ph.D.) at the IDS and CICD of NTUB (国立台北商业大学资讯与决策科学研究所暨智能控制与决策研究室教授); at the ME Dept. and CAIDS of MCUT (2020~2022借调至明志科技大学机械工程系任特聘教授兼人工智慧暨资料科学研究中心主任); the CSQ (2019年起任中华民国品质学会大数据品质应用委员会主任委员); the DSBA (2013年创立台湾资料科学与商业应用协会); and the CARS (2012年创立中华R软体学会) 
Notes: This code is provided without warranty.
'''

#### 3.3.1 重抽样与资料切分方法
#### p.333 前半段
import pandas as pd
# import numpy as np
from resample import bootstrap # !pip install resample==0.21 or !conda install -c conda-forge resample --y (旧版Python用前者，新版用后者) 0.21 (o) -> 1.0.1 (x)
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

airquality = pd.read_csv('airquality.csv', encoding='utf-8')
# 了解airquality 的资料结构
airquality.info()

# 分别视觉化Ozon 与Wind 和Temp 的关系
plt.subplot(121) # 1: 一横，2: 两纵，1: 现在画第一张图
plt.scatter(airquality["Wind"],airquality["Ozone"])
plt.xlabel('Wind')
plt.ylabel('Ozone')

plt.subplot(122) # 1: 一横，2: 两纵，2: 现在画第二张图
plt.scatter(airquality["Temp"],airquality["Ozone"])
plt.xlabel('Temp')
# plt.close()

# 以各变项中位数填补遗缺值
cleanAQ = pd.DataFrame()
for col in airquality.columns:
    cleanAQ[col] = airquality[col].fillna(airquality[col].median())

# 确认资料表中已无遗缺值
cleanAQ.isnull().sum()

# https://github.com/dsaxton/resample/blob/master/examples/resample.ipynb
# 定义拔靴抽样函数bootstrap() 所用的统计值计算函数fitreg()(类似R语言中的rsq())
def fitreg(A): # A表拔靴抽样的样本子集
    X = A[:,2:4] # select Wind and Temp 选择 Wind 和Temp(注意前包后不包！)
    y = A[:,0] # select Ozon 选择
    lm = LinearRegression()
    lm.fit(X, y)
    r_squared = lm.score(X, y) # Return the coefficient of determination of the prediction
    return {"rsq": r_squared}

# cleanAQ - bootstrapping -> 1000 resamples (sample size ? Ans. 153 same as original sample size) -> Fit LR models -> 1000 R^2 -> central tendency -> dispersion (variance or standard deviation)
# boot_coef = bootstrap.bootstrap(a=cleanAQ.values, f=fitreg, b=1000) # array of dict (从a进行b次拔靴抽样，每次抽出之样本进行f中定义的配适工作)
boot_coef = bootstrap.bootstrap(fitreg, cleanAQ.values, size=1000) # array of dict (从a进行b次拔靴抽样，每次抽出之样本进行f中定义的配适工作)
# 将boot_coef字典转成dataframe
param_frame = pd.DataFrame([pd.Series(x) for x in boot_coef])
param_frame.head()
param_frame.shape # (1000, 1)
param_frame.hist()

# Python中如何绘制常态分位数图(Normal Probability Plots)
import statsmodels.api as sm
from matplotlib import pyplot as plt
fig = sm.qqplot(param_frame.rsq, line="q") # Attention to line="q" and the default comparison distribution is scipy.stats.distributions.norm (a standard normal).
plt.show()

# p.341
# Sklearn中的CV与KFold详解
# https://blog.csdn.net/FontThrone/article/details/79220127
X = pd.read_csv('predictors.csv', encoding='utf-8')
y = pd.read_csv('classes.csv', encoding='utf-8')

#### 3.3.1 后半段
# 简单保留法
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 

# 交叉验证法
from sklearn.model_selection import StratifiedKFold
stratified_folder = StratifiedKFold(n_splits=10, random_state=0, shuffle=True) # Try StratifiedKFold(n_splits=10, shuffle=False)

# Please double click the stratified_folder object and be attentive to the method split
# 回圈搭配split()方法检视十折交叉验证样本编号
for train_index, test_index in stratified_folder.split(X_train, y_train): # Generate indices to split data into training and test set.
    print("Stratified Train Index:", train_index)
    print("Stratified Test Index:", test_index)
    

#### 3.3.2 单类模型参数调校 & 3.3.2.1 多个参数待调 
# p.347
# https://blog.csdn.net/luanpeng825485697/article/details/79831703
import pandas as pd
Smarket = pd.read_csv('Smarket.csv', encoding='utf-8')

# k近邻法不允许遗缺值
Smarket.isnull().sum()
Smarket.Year.value_counts()

# S&P Stock Market Data: Daily percentage returns for the S&P 500 stock index between 2001 and 2005. The goal is to predict whether the index will increase or decrease on a given day using the past 5 days’ percentage changes in the index.
X = Smarket.iloc[:,1:8] # 前包后不包
y = Smarket.iloc[:,8]

# 校验集与测试集切分
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1234, test_size=0.25) # 校验集与最终测试集

# 校验集属性矩阵与前处理(i.e. kNN因欧几里德直线距离需标准化各个变量)
from sklearn import preprocessing
X_train = preprocessing.scale(X_train) # Python相对少见的泛函式编程语法

from sklearn.model_selection import cross_val_score, GridSearchCV # 参数grid下，以CV search最佳超参数，cross_val_score负责评分运算
from sklearn.neighbors import KNeighborsClassifier

# 最佳化参数k的调校范围
import numpy as np
k_range = np.arange(start=5,stop=45,step=2)
# 待估参数权重的取值范围。 uniform为统一取权值，distance表示距离倒数取权值(与R不同之处在于多调了一个参数weight_options！)
weight_options = ['uniform', 'distance'] # 各邻居票票等值 versus 依距离的倒数来加权 
# 下面构建parameter grid，其结构是key为参数名称，value是待搜索的数值列表的一个字典结构
# 定义最佳化参数字典，字典中的key值必须是分类算法的函数的参数名
param_grid = {'n_neighbors': k_range, 'weights': weight_options} 
print(param_grid)

# 定义分类算法。 n_neighbors和weights的参数名称和param_grid字典中的key名对应
knn = KNeighborsClassifier(n_neighbors=5) # number of neighbors ca be specified arbitrarily

# 这里GridSearchCV的参数形式和cross_val_score的形式差不多，其中param_grid是parameter grid所对应的参数
# GridSearchCV中的n_jobs设置为-1时，可以实现并行计算（如果你的电脑支持的情况下，-1 means using all processors.）
#针对每个参数对进行了10次交叉验证。 scoring='accuracy'使用准确率为结果的度量指标。可以添加多个度量指标
grid = GridSearchCV(estimator = knn, param_grid = param_grid, cv=10, scoring='accuracy') # grid的空模
import time
start = time.time()
grid.fit(X_train, y_train) # grid已变成实模
end = time.time()
end - start

# 结果都藏在实模grid的后面了！
dir(grid)

print('分数纪录：',grid.cv_results_)
print('最佳分数:',grid.best_score_)  
print('最佳参数：',grid.best_params_)
print('最佳模型：',grid.best_estimator_)  

grid.cv_results_
type(grid.cv_results_) # A dict
grid.cv_results_.keys()

# 交叉验证结果字典，转换成DataFrame
cv_res = pd.DataFrame(grid.cv_results_)

# 验证交叉验证绩效评估计算结果
cv_res.iloc[0, 7:17].mean() # 0.8783916723861817 每次结果不同！
cv_res.iloc[0, 7:17].std(ddof=0) # 0.030754122995618343 (pandas Series预设变异数与标准差计算之ddof=1(分母n-1)，不过sklearn却用ddof=0(分母n))

# 使用获取的最佳参数，搭配全部的核验集样本937个，重新训练模型，预测测试样本标签(图3.15的哪个block?)
knn = KNeighborsClassifier(n_neighbors=grid.best_params_['n_neighbors'], weights=grid.best_params_['weights'])
knn.fit(X_train, y_train) # 倒进全部核验集/训练集样本，配适最佳模型 knn

# 最终绩效评定 model assessment (图3.14的哪个block?)
knn.score(X_test, y_test)
test_pred = knn.predict(X_test)

#### 自行完成最终测试绩效评估工作
# Homewrok due date on August/12th 17:00: y_test, test_pred 产生混淆矩阵，计算整体指标与类别指标
from sklearn.metrics import classification_report
temp = classification_report(y_test, test_pred)
type(temp) # a str easy to check results

#### 3.3.3 比较不同类的模型(与书本/讲义上的范例不同，且假设各算法预设参数可获得最佳模型！)

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
models = [] # tuple ('简记名', 建模类别函数全名)
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# evaluate each model in turn
results = [] # 逐次添加各模型交叉验证绩效评量结果
names = []
scoring = 'balanced_accuracy'
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True) # 十折交叉验证样本切分
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

# 可自行完成统计检定：ANOVA检定检视是否有显著的差异

import scipy.stats as stats

fvalue, pvalue = stats.f_oneway(results[0], results[1], results[2], results[3], results[4], results[5])

print("检定统计量：{}，检定p值：{}".format(fvalue, pvalue))

# 后续可以Tukey HSD找出正确率最大方法

dfn, dfd = 5, 55
stats.f.sf(fvalue,dfn,dfd)

# P(F>3.7728334504659156)=0.0052222146020425185 (p-value < 0.05) 每次结果不尽相同
# Therefore, the hypothesis H0: mean1 = mean2 = ... = mean6 is rejected.
# That is, at least one of these 6 models is better than others. 拒绝各模型无差异的虚无假说，可能至少有ㄧ个模型比其他模型表现得更好。

#### Reference: How To Compare Machine Learning Algorithms in Python with scikit-learn (https://machinelearningmastery.com/compare-machine-learning-algorithms-python-scikit-learn/)