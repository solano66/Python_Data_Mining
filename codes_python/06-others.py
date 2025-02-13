'''
Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the IDS, CICD, and CISD of NTUB (臺北商業大學資訊與決策科學研究所暨智能控制與決策研究室教授兼校務永續發展中心主任); at the ME Dept. and CAIDS of MCUT (2020年借調至明志科技大學機械工程系擔任特聘教授兼人工智慧暨資料科學研究中心主任兩年); the CSQ (2019年起任中華民國品質學會大數據品質應用委員會主任委員); the DSBA (2013年創立臺灣資料科學與商業應用協會); and the CARS (2012年創立中華R軟體學會)
Notes: This code is provided without warranty.
'''

#### 6.1.2.1 房價中位數預測案例
## ------------------------------------------------------------------------
# 載入Boston 房價資料匯入方法load_boston()
from sklearn.datasets import load_boston
boston = load_boston()

print(type(boston))

# Bunch 物件的鍵
print(boston.keys())
# print(boston.DESCR)

# Python 句點語法檢視屬性矩陣data 與目標變數target 維度與維數
print(boston.data.shape)

print(boston.target.shape)

# 13 個屬性名稱
print(boston.feature_names[:9])

print(boston.feature_names[9:])

# Bunch 物件轉為DataFrame
import pandas as pd
data = pd.DataFrame(boston.data)
data.columns = boston.feature_names

# 以上load_boston()無法載入時執行下兩行
# import pandas as pd
# data = pd.read_csv("./_data/Boston.csv")

data.medv.value_counts()

print(data.head())

# 添加目標變數於後
data['PRICE'] = boston.target

# DataFrame 的info() 方法
print(data.info())

# 摘要統計表
print(data.describe(include='all'))

# 載入建模套件與績效評估類別
import xgboost as xgb # conda install xgboost --y; python -m pip install --upgrade pip setuptools wheel; /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)" or /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"; brew install libomp
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# 切分屬性矩陣與目標變數
X, y = data.iloc[:,:-1], data.iloc[:,-1]
# xgboost 套件的資料結構DMatrix (後面計算量大交叉驗證使用)
data_dmatrix = xgb.DMatrix(data=X,label=y)
# xgboost.core.DMatrix
print(type(data_dmatrix))

# 切分訓練集(80%) 與測試集(20%)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.2, random_state=123)
# 宣告xgboost 迴歸模型規格
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror',
colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, # boosting max_depth, how about bagging?
alpha = 10, n_estimators = 10) # objective ='reg:linear' is deprecated

# 傳入資料擬合模型
xg_reg.fit(X_train,y_train)
# 預測測試集資料
preds = xg_reg.predict(X_test)
# 傳入實際值與預測值向量計算均方根誤差(一萬元左右)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE 為%f" % (rmse))

# 訓練參數同前
params = {"objective":"reg:squarederror",'colsample_bytree': 0.3,
'learning_rate': 0.1, 'max_depth': 5, 'alpha': 10,
'silent': 1} # objective ='reg:linear' is deprecated
# k 摺交叉驗證訓練XGBoost
cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3, num_boost_round=50, early_stopping_rounds=10,
metrics="rmse", as_pandas=True, seed=123, verbose_eval=False)

# 三次交叉驗證計算訓練集與測試集RMSE 的平均數和標準差
print(cv_results.head(15))

# 50 回合(橫列編號) 的XGBoost 訓練與測試，RMSE 平均值逐回降低
print(cv_results.tail())

# 最後一回合XGBoost 測試集RMSE 的平均值
print(cv_results["test-rmse-mean"].tail(1))

# 訓練與預測回合數訂為10
xg_reg = xgb.train(params=params, dtrain=data_dmatrix,
num_boost_round=10)
# XGBoost 變數重要度繪圖
import matplotlib.pyplot as plt
ax = xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [5, 5]
# plt.show()
fig = ax.get_figure()
# fig.savefig('./_img/importance.png')

#### 6.2.8 深度學習參數調校
## ------------------------------------------------------------------------
import sys # 系統相關參數與函式模組，是一個強大的Python 標準函式庫
print (sys.version) # 直譯器版本號與所使用的編譯器

import numpy
# 載入模型選擇模組中重要的交叉驗證網格調參類別
from sklearn.model_selection import GridSearchCV
# Python 深度學習友善API Keras
from keras.models import Sequential
from keras.layers import Dense # 載入神經網路稠密連通層
from keras.wrappers.scikit_learn import KerasClassifier

def create_model():
    # 建立單層隱藏層神經網路模型
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # 設定損失函數、優化算法與績效衡量
    model.compile(loss='binary_crossentropy', optimizer=
    'adam', metrics=['accuracy'])
    return model

# 讀入資料檔與建立多層感知機模型
#path = '/Users/Vince/cstsouMac/Python/Examples/DeepLearning/'
#fname = 'py_codes/data/pima-indians-diabetes.csv'
path = ''
fname = '_data/pima-indians-diabetes.csv'
dataset =numpy.loadtxt(''.join([path, fname]), delimiter=",")
X = dataset[:,0:8]
y = dataset[:,8]
model = KerasClassifier(build_fn=create_model, verbose=0)

# 建立待調參數網格字典
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
param_grid = dict(batch_size=batch_size, epochs=epochs)
# 宣告交叉驗證網格調參模型物件
grid = GridSearchCV(estimator=model, param_grid=param_grid)

seed = 7 # for the reproducibility results
numpy.random.seed(seed)
import time
start = time.time()
# 傳入資料進行調參配適
grid_result = grid.fit(X, y)
end = time.time()
print(end - start)

print(type(grid_result))
# print(dir(grid_result)) # 請自行執行

# 檢視參數調校的最佳結果
print("Best: %f using %s" % (grid_result.best_score_,
grid_result.best_params_))

# 以迴圈印出所有調參結果
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

