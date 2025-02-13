'''
Collated by Prof. Ching-Shih Tsou 邹庆士 教授 (Ph.D.) at the IDS, CICD, and CISD of NTUB (台北商业大学资讯与决策科学研究所暨智能控制与决策研究室教授兼校务永续发展中心主任); at the ME Dept. and CAIDS of MCUT (2020年借调至明志科技大学机械工程系担任特聘教授兼人工智慧暨资料科学研究中心主任两年); the CSQ (2019年起任品质学会大数据品质应用委员会主任委员); the DSBA (2013年创立台湾资料科学与商业应用协会); and the CARS (2012年创立中华R软体学会)
Notes: This code is provided without warranty.
'''

#### 6.1.2.1 房价中位数预测案例
## ------------------------------------------------------------------------
# 载入Boston 房价资料汇入方法load_boston()
from sklearn.datasets import load_boston
boston = load_boston()
print(type(boston))

# Bunch 物件的键
print(boston.keys())
# print(boston.DESCR)

# Python 句点语法检视属性矩阵data 与目标变数target 维度与维数
print(boston.data.shape)

print(boston.target.shape)

# 13 个属性名称
print(boston.feature_names[:9])

print(boston.feature_names[9:])

# Bunch 物件转为DataFrame
import pandas as pd
data = pd.DataFrame(boston.data)
data.columns = boston.feature_names
print(data.head())

# 添加目标变数于后
data['PRICE'] = boston.target

# DataFrame 的info() 方法
print(data.info())

# 摘要统计表
print(data.describe(include='all'))

# 载入建模套件与绩效评估类别
import xgboost as xgb # conda install xgboost --y; python -m pip install --upgrade pip setuptools wheel; /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)" or /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"; brew install libomp
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# 切分属性矩阵与目标变数
X, y = data.iloc[:,:-1], data.iloc[:,-1]
# xgboost 套件的资料结构DMatrix (后面计算量大交叉验证使用)
data_dmatrix = xgb.DMatrix(data=X,label=y)
# xgboost.core.DMatrix
print(type(data_dmatrix))

# 切分训练集(80%) 与测试集(20%)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.2, random_state=123)
# 宣告xgboost 回归模型规格
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror',
colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, # boosting max_depth, how about bagging?
alpha = 10, n_estimators = 10) # objective ='reg:linear' is deprecated

# 传入资料拟合模型
xg_reg.fit(X_train,y_train)
# 预测测试集资料
preds = xg_reg.predict(X_test)
# 传入实际值与预测值向量计算均方根误差(一万元左右)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE 为%f" % (rmse))

# 训练参数同前
params = {"objective":"reg:squarederror",'colsample_bytree': 0.3,
'learning_rate': 0.1, 'max_depth': 5, 'alpha': 10,
'silent': 1} # objective ='reg:linear' is deprecated
# k 折交叉验证训练XGBoost
cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3, num_boost_round=50, early_stopping_rounds=10,
metrics="rmse", as_pandas=True, seed=123, verbose_eval=False)

# 三次交叉验证计算训练集与测试集RMSE 的平均数和标准差
print(cv_results.head(15))

# 50 回合(横列编号) 的XGBoost 训练与测试，RMSE 平均值逐回降低
print(cv_results.tail())

# 最后一回合XGBoost 测试集RMSE 的平均值
print(cv_results["test-rmse-mean"].tail(1))

# 训练与预测回合数订为10
xg_reg = xgb.train(params=params, dtrain=data_dmatrix,
num_boost_round=10)
# XGBoost 变数重要度绘图
import matplotlib.pyplot as plt
ax = xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [5, 5]
# plt.show()
fig = ax.get_figure()
# fig.savefig('./_img/importance.png')

#### 6.2.8 深度学习参数调校
## ------------------------------------------------------------------------
import sys # 系统相关参数与函式模组，是一个强大的Python 标准函式库
print (sys.version) # 直译器版本号与所使用的编译器

import numpy
# 载入模型选择模组中重要的交叉验证网格调参类别
from sklearn.model_selection import GridSearchCV
# Python 深度学习友善API Keras
from keras.models import Sequential
from keras.layers import Dense # 载入神经网路稠密连通层
from keras.wrappers.scikit_learn import KerasClassifier

def create_model():
    # 建立单层隐藏层神经网路模型
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # 设定损失函数、优化算法与绩效衡量
    model.compile(loss='binary_crossentropy', optimizer=
    'adam', metrics=['accuracy'])
    return model

# 读入资料档与建立多层感知机模型
#path = '/Users/Vince/cstsouMac/Python/Examples/DeepLearning/'
#fname = 'py_codes/data/pima-indians-diabetes.csv'
path = ''
fname = '_data/pima-indians-diabetes.csv'
dataset =numpy.loadtxt(''.join([path, fname]), delimiter=",")
X = dataset[:,0:8]
y = dataset[:,8]
model = KerasClassifier(build_fn=create_model, verbose=0)

# 建立待调参数网格字典
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
param_grid = dict(batch_size=batch_size, epochs=epochs)
# 宣告交叉验证网格调参模型物件
grid = GridSearchCV(estimator=model, param_grid=param_grid)

seed = 7 # for the reproducibility results
numpy.random.seed(seed)
import time
start = time.time()
# 传入资料进行调参配适
grid_result = grid.fit(X, y)
end = time.time()
print(end - start)

print(type(grid_result))
# print(dir(grid_result)) # 请自行执行

# 检视参数调校的最佳结果
print("Best: %f using %s" % (grid_result.best_score_,
grid_result.best_params_))

# 以回圈印出所有调参结果
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

