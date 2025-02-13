'''
Collated by Ching-Shih Tsou 鄒慶士 博士 (Ph.D.) Distinguished Prof. at the Department of Mechanical Engineering/Director at the Center of Artificial Intelligence & Data Science (機械工程系特聘教授兼人工智慧暨資料科學研究中心主任), MCUT (明志科技大學); Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), NTUB (國立臺北商業大學)
Notes: This code is provided without warranty.
'''


#### 薈萃式學習 Ensemble Learning

#### 數據匯入與前處理
## from sklearn import datasets
import numpy as np
import pandas as pd # /Users/Vince/anaconda

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

from sklearn.ensemble import GradientBoostingRegressor  # 集成方法迴歸套件：梯度效能提升迴歸樹
from sklearn.model_selection import GridSearchCV  # 載入交叉驗證套件：網格參數最佳化交叉驗證(update sklearn to 0.19.1, scipy and restart Python!)
import matplotlib.pyplot as plt  # 載入繪圖套件

#import os
#os.chdir('/Users/Vince/cstsouMac/Python/Examples/NHRI/FIA_III_Python_DM_0506_2018_codes')

raw_data = pd.read_csv('./_data/products_sales.txt', delimiter=',')

#### Basic info. of dataset
raw_data.head()

print ('{:*^60}'.format('Data overview:'))
print (raw_data.tail(2))  # 印出原始數據後2條
print ('{:*^60}'.format('Data dtypes:'))
print (raw_data.dtypes)  # 印出資料類型
print ('{:*^60}'.format('Data DESC:'))
print (raw_data.describe().round(1).T)  # 印出原始資料基本描述性統計信息

# limit_info: 是否限購 [0  1 10??]
# campaign_type: 促銷活動類型 [0 1 2 3 4 5 6]
# campaign_level: 促銷活動重要性 [0 1]
# product_level: 產品重要度 [1 2 3]
# resource_amount: 促銷資源數量
# email_rate: 電郵促銷含該產品的比例
# price: 單價
# discount_rate: 折扣率
# hour_resources: 促銷展示時數
# campaign_fee: 產品促銷綜合費用
# orders: 每次活動的訂單量

col_names = ['limit_infor', 'campaign_type', 'campaign_level', 'product_level']  # 定義要查看的行column

#### Freq. distribution of first five categorical variables
for col_name in col_names:  # 使用迴圈讀取每行column
    unque_value = np.sort(raw_data[col_name].unique())  # 獲得行column的唯一值
    print ('{:*^50}'.format('{1} unique values:{0}').format(unque_value, col_name))  # 印出

for col_name in col_names:  # 使用迴圈讀取每行column
    vc = raw_data[col_name].value_counts()  # 獲得行column的次數分配表
    print ('\n{:*^50}'.format('freq table of {1} :\n{0}').format(vc, col_name))  # 印出

raw_data.loc[:, col_names].apply(lambda x: x.value_counts(), axis=0).T.stack()

# raw_data.limit_infor.value_counts()

na_cols = raw_data.isnull().any(axis=0)  # 查看每一行column是否具有遺缺值
print ('{:*^60}'.format('NA Cols:'))
print (na_cols)  # 查看具有遺缺值的行column
na_lines = raw_data.isnull().any(axis=1)  # 查看每一列row是否具有遺缺值
print ('Total number of NA lines is: {0}'.format(na_lines.sum()))  # 查看具有遺缺值的總列row數

print ('{:*^60}'.format('Correlation Analyze:'))
short_name = ['li', 'ct', 'cl', 'pl', 'ra', 'er', 'price', 'dr', 'hr', 'cf', 'orders']
long_name = raw_data.columns
name_dict = dict(zip(long_name, short_name))
print (raw_data.corr().round(2).rename(index=name_dict, columns=name_dict))  # 輸出所有輸入特徵變量以及預測變量的相關性矩陣
print (name_dict)

# help(pd.core.frame.DataFrame.rename)

#- 數據預處理，遺缺值替換為平均值後儲存為sales_data (fill.na方法)
#- 只保留促銷值為0和1的記錄(促銷值10的紀錄只有一筆)
#- 異常值處理，將異常極大值替換為平均值 (replace方法)
#- 印出處理完成數據基本描述性信息 (describe + rename方法)

sales_data = raw_data.fillna(raw_data['price'].mean())  # 遺缺值替換為平均值
# sales_data = raw_data.drop('email_rate',axis=1) # 或移除遺缺值
sales_data = sales_data[sales_data['limit_infor'].isin((0, 1))]  # 只保留促銷值為0和1的記錄(logical indexing)
sales_data['campaign_fee'] = sales_data['campaign_fee'].replace(33380, sales_data['campaign_fee'].mean())  # 將異常極大值替換為平均值
print ('{:*^60}'.format('transformed data:'))
print (sales_data.describe().round(2).T.rename(index=name_dict))  # 印出處理完成數據基本描述性信息

X = sales_data.iloc[:, :-1]  # 分割X(從頭取到倒數第一column)
y = sales_data.iloc[:, -1]  # 分割y(取倒數第一column)

from sklearn.model_selection import train_test_split # cross_validation has been changed to model_selection

X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train) # 依X_train估計mu與sigma

#sc.mean_
#sc.scale_

X_train_std = sc.transform(X_train) # 真正作轉換
X_test_std = sc.transform(X_test) # 以X_train的mu與sigma對X_test作轉換

print (sc)
# print (help(sc))


#### 基於超參數最佳化的Gradient Boosting的銷售預測模型
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
model_gbr = GradientBoostingRegressor()  # 建立GradientBoostingRegressor迴歸物件
parameters = {'loss': ['ls', 'lad', 'huber', 'quantile'],
              'min_samples_leaf': [1, 2, 3, 4, 5],
              'alpha': [0.1, 0.3, 0.6, 0.9]}  # 定義要優化的參數信息之字典(alphafloat, default=0.9. The alpha-quantile of the huber loss function and the quantile loss function. Only if loss='huber' or loss='quantile'. Values must be in the range (0.0, 1.0).)
model_gs = GridSearchCV(estimator=model_gbr, param_grid=parameters, cv=5)  # 建立交叉驗證模型對象

import time
start = time.time()
"the code you want to test stays here"
model_gs.fit(X, y)  # 訓練交叉驗證模型(所以丟全部樣本進去)
end = time.time()
print(end - start) # 37.64466881752014

print ('Best score is:', model_gs.best_score_)  # 獲得交叉驗證模型得出的最佳得分
print ('Best parameter is:', model_gs.best_params_)  # 獲得交叉驗證模型得出的最佳參數

type(model_gs)
dir(model_gs)

model_best = model_gs.best_estimator_  # 獲得交叉驗證模型得出的最佳模型物件
model_best
model_best.fit(X, y)  # 訓練最佳模型
plt.style.use("ggplot")  # 應用ggplot樣式
plt.figure()  # 建立畫布物件
plt.plot(np.arange(X.shape[0]), y, label='true y')  # 畫出原始變量的曲線
plt.plot(np.arange(X.shape[0]), model_best.predict(X), label='predicted y')  # 畫出預測變量的曲線
plt.legend(loc=0)  # 設置圖例位置
plt.show()  # 顯示圖形

New_X = np.array([[1, 1, 0, 1, 15, 0.5, 177, 0.66, 101, 798]])  # 要預測的新數據記錄(10個變數)
print ('{:*^60}'.format('Predicted orders:'))
print (model_best.predict(New_X).round(0))  # 印出預測值


# 拔靴集成法
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier.predict_proba
#model_BagC = BaggingClassifier(n_estimators=20, random_state=0)  # Bagging分類模型對象
#
#start = time.time()
#model_BagC.fit(X, y)  # 訓練交叉驗證模型
#end = time.time()
#print(end - start) # 1.1966552734375
#
#dir(model_BagC)
#model_BagC.n_classes_ # 694, why?
#y.value_counts()

#print ('{:*^60}'.format('Predicted orders:'))
#print (model_BagC.predict(New_X).round(0))  # 印出輸出預測值

#### 拔靴集成法Bagging與隨機森林Random Forest的集成模型
import numpy as np  # numpy套件
import pandas as pd  # pandas套件
from sklearn.feature_extraction import DictVectorizer  # 數值分類轉整數分類套件
from imblearn.over_sampling import SMOTE  # 過度抽樣處理套件SMOTE(!conda install -c conda-forge imbalanced-learn --y)
from sklearn.model_selection import StratifiedKFold, cross_val_score  # 導入交叉驗證算法
from sklearn.linear_model import LogisticRegression  # 導入邏輯斯迴歸套件
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, BaggingClassifier  # 三種集成分類套件和投票方法套件：投票分類器、隨機森林分類器、拔靴集成分類器


# 數據審查和預處理函數
# 基本狀態查看
def set_summary(df):
    '''
    查看數據集後2條數據、數據類型、描述性統計
    :param df: 數據框
    :return: 無
    '''
    print ('{:*^60}'.format('Data overview:'))
    print (df.tail(2))  # 打印原始數據後2條
    print ('{:*^60}'.format('Data dtypes:'))
    print (df.dtypes)  # 打印數據類型
    print ('{:*^60}'.format('Data DESC:'))
    print (df.describe().round(2).T)  # 打印原始數據基本描述性信息


# 缺失值審查
def na_summary(df):
    '''
    查看數據集的缺失數據列、行記錄數
    :param df: 數據框
    :return: 無
    '''
    na_cols = df.isnull().any(axis=0)  # 查看每一行是否具有缺失值
    print ('{:*^60}'.format('NA Cols:'))
    print (na_cols)  # 查看具有缺失值的行
    na_lines = df.isnull().any(axis=1)  # 查看每一行是否具有缺失值
    print ('Total number of NA lines is: {0}'.format(na_lines.sum()))  # 查看具有缺失值的行總記錄數


# 類樣本均衡審查
def label_samples_summary(df):
    '''
    查看每個類的樣本量分布
    :param df: 數據框
    :return: 無
    '''
    print ('{:*^60}'.format('Labels samples count:'))
    print (df.iloc[:, 1].groupby(df.iloc[:, -1]).count()) # df.iloc[:,-1].value_counts()


# 字符串分類轉整數分類
def str2int(set, convert_object, unique_object, training=True):
    '''
    用於將分類變量中的字符串轉換為數值索引分類
    :param set: 數據集
    :param convert_object:  DictVectorizer轉換對象，當training為True時為空；當training為False時則使用從訓練階段得到的對象
    :param unique_object: 唯一值列表，當training為True時為空；當training為False時則使用從訓練階段得到的唯一值列表
    :param training: 是否為訓練階段
    :return: 訓練階段返回model_dvtransform,unique_list,traing_part_data；預測應用階段返回predict_part_data
    '''
    convert_cols = ['cat', 'attribution', 'pro_id', 'pro_brand', 'order_source', 'pay_type', 'use_id',
                    'city']  # 定義要轉換的行
    final_convert_matrix = set[convert_cols]  # 獲得要轉換的數據集合
    lines = set.shape[0]  # 獲得總記錄數
    dict_list = []  # 總空列表，用於存放字符串與對應索引組成的字典
    if training == True:  # 如果是訓練階段
        unique_list = []  # 總唯一值列表，用於存儲每個行的唯一值列表
        for col_name in convert_cols:  # 循環讀取每個行名
            cols_unqiue_value = set[col_name].unique().tolist()  # 獲取行的唯一值列表
            unique_list.append(cols_unqiue_value)  # 將唯一值列表追加到總列表
        for line_index in range(lines):  # 讀取每行索引
            each_record = final_convert_matrix.iloc[line_index]  # 獲得每行數據，是一個Series
            for each_index, each_data in enumerate(each_record):  # 讀取Series每行對應的索引值
                list_value = unique_list[each_index]  # 讀取該行索引對應到總唯一值列表列索引下的數據(其實是相當於原來的行做了轉置成了列，目的是查找唯一值在列表中的位置)
                each_record[each_index] = list_value.index(each_data)  # 獲得每個值對應到總唯一值列表中的索引
            each_dict = dict(zip(convert_cols, each_record))  # 將每個值和對應的索引組合字典
            dict_list.append(each_dict)  # 將字典追加到總列表
        model_dvtransform = DictVectorizer(sparse=False, dtype=np.int64)  # 建立轉換模型對象
        model_dvtransform.fit(dict_list)  # 應用分類轉換訓練
        traing_part_data = model_dvtransform.transform(dict_list)  # 轉換訓練集
        return model_dvtransform, unique_list, traing_part_data
    else:  # 如果是預測階段
        for line_index in range(lines):  # 讀取每行索引
            each_record = final_convert_matrix.iloc[line_index]  # 獲得每行數據，是一個Series
            for each_index, each_data in enumerate(each_record):  # 讀取Series每行對應的索引值
                list_value = unique_object[each_index]  # 讀取該行索引對應到總唯一值列表列索引下的數據(其實是相當於原來的行做了轉置成了列，目的是查找唯一值在列表中的位置)
                each_record[each_index] = list_value.index(each_data)  # 獲得每個值對應到總唯一值列表中的索引
            each_dict = dict(zip(convert_cols, each_record))  # 將每個值和對應的索引組合字典
            dict_list.append(each_dict)  # 將字典追加到總列表
        predict_part_data = convert_object.transform(dict_list)  # 轉換預測集
        return predict_part_data


# 時間屬性拓展(Convert map object to numpy array in python 3: https://stackoverflow.com/questions/28524378/convert-map-object-to-numpy-array-in-python-3)
def datetime2int(set):
    '''
    將日期和時間數據拓展出其他屬性，例如星期幾、周幾、小時、分鐘等。
    :param set: 數據集
    :return: 拓展後的屬性矩陣
    '''
    date_set = map(lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d'),
                   set['order_date'])  # 將set中的order_date(年月日)行轉換為特定日期格式
    date_set = np.array(list(date_set))
    weekday_data = map(lambda data: data.weekday(), date_set)  # 周幾
    weekday_data = np.array(list(weekday_data))
    daysinmonth_data = map(lambda data: data.day, date_set)  # 當月幾號
    daysinmonth_data = np.array(list(daysinmonth_data))
    month_data = map(lambda data: data.month, date_set)  # 月份
    month_data = np.array(list(month_data))

    time_set = map(lambda times: pd.datetime.strptime(times, '%H:%M:%S'),
                   set['order_time'])  # 將set中的order_time(時分秒)行轉換為特定時間格式
    time_set = np.array(list(time_set))
    second_data = map(lambda data: data.second, time_set)  # 秒
    second_data = np.array(list(second_data))
    minute_data = map(lambda data: data.minute, time_set)  # 分鐘
    minute_data = np.array(list(minute_data))
    hour_data = map(lambda data: data.hour, time_set)  # 小時
    hour_data = np.array(list(hour_data))

    final_set = []  # 列表，用於將上述拓展屬性組合起來
    final_set.extend((weekday_data, daysinmonth_data, month_data, second_data, minute_data, hour_data))  # 將屬性列表批量組合
    final_matrix = np.array(final_set).T  # 轉換為矩陣並轉置
    return final_matrix


# 樣本均衡
def sample_balance(X, y):
    '''
    使用SMOTE方法對不均衡樣本做過抽樣處理
    :param X: 輸入特徵變量X
    :param y: 目標變量y
    :return: 均衡後的X和y
    '''
    model_smote = SMOTE()  # 建立SMOTE模型對象
    x_smote_resampled, y_smote_resampled = model_smote.fit_sample(X, y)  # 輸入數據並作過抽樣處理
    return x_smote_resampled, y_smote_resampled


# 數據套用
# 定義特殊字段數據格式
dtypes = {'order_id': np.object,
          'pro_id': np.object,
          'use_id': np.object}
raw_data = pd.read_csv('./data/abnormal_orders.txt', delimiter=',', dtype=dtypes)  # 讀取數據集

# 數據審查
set_summary(raw_data)  # 基本狀態查看
na_summary(raw_data)  # 缺失值審查
label_samples_summary(raw_data)  # 類樣本分布審查

#訂單編號
#訂單日期
#訂單時間
#商品類別
#商品來源通路
#商品編號
#商品品牌
#商品銷售金額
#商品銷售數量
#訂單來源通路
#支付類型
#客戶編號
#下單城市

# 數據預處理
drop_na_set = raw_data.dropna()  # 丟棄帶有NA值的數據row(134190 -> 132761)
X_raw = drop_na_set.iloc[:, 1:-1]  # 分割輸入變量X，並丟棄訂單ID column和最後的目標變量
y_raw = drop_na_set.iloc[:, -1]  # 分割目標變量y

# X_raw.dtypes
#X_raw.columns[[0, 1, 2, 3, 4, 5, 8, 9, 10, 11]]
#tbls = X_raw[X_raw.columns[[0, 1, 2, 3, 4, 5, 8, 9, 10, 11]]].apply(lambda x: np.unique(x.values, return_counts=True))
#tbls.index
#tbls.order_date # 2013整年資料
#tbls.order_time
#tbls['cat'] # tbls.cat not working !!
#tbls.attribution
#tbls.pro_id
#tbls.pro_brand
#tbls.order_source
#tbls.pay_type
#tbls.use_id
#tbls.city
#
#X_raw.attribution.value_counts()
#np.unique(X_raw.attribution.values, return_counts=True)

### 各類別變數次數分配表
#tbl2 =[]
#for col in X_raw.columns[[0, 1, 2, 3, 4, 5, 8, 9, 10, 11]]:
#    tbl2.append(X_raw[col].value_counts()) # X_raw.col is wrong.

import time
start = time.time()
model_dvtransform, unique_object, str2int_data = str2int(X_raw, None, None, training=True)  # 字符串分類轉整數型分類(str2int_data: 132761*8)
end = time.time()
print(end - start) # 165.65647315979004

datetime2int_data = datetime2int(X_raw)  # 拓展日期時間屬性(132761, 6 週日月秒分時)
combine_set = np.hstack((str2int_data, datetime2int_data))  # 合併轉換後的分類和拓展後的日期數據集(132761, 14)
constant_set = X_raw[['total_money', 'total_quantity']]  # 原始連續數據變量(132761, 2)
X_combine = np.hstack((combine_set, constant_set))  # 再次合併數據集(132761, 16)
#np.savetxt("./data/abnormal_orders.csv", X_combine, delimiter=",")
feature_name = ['attribution', 'cat', 'city', 'order_source', 'pay_type', 'pro_brand', 'pro_id', 'use_id', 'week', 'day', 'month', 'sec', 'min', 'hr', 'total_money', 'total_quantity']
X_combine = pd.read_csv("./data/abnormal_orders.csv", header=None, names=feature_name)
X, y = sample_balance(X_combine, y_raw)  # 樣本均衡處理

# 組合分類模型交叉驗證
model_rf = RandomForestClassifier(n_estimators=20, random_state=0)  # 隨機森林分類模型對象
model_lr = LogisticRegression(random_state=0)  # 邏輯迴歸分類模型對象
model_BagC = BaggingClassifier(n_estimators=20, random_state=0)  # Bagging分類模型對象
estimators = [('randomforest', model_rf), ('Logistic', model_lr), ('bagging', model_BagC)]  # 建立組合評估器列表
model_vot = VotingClassifier(estimators=estimators, voting='soft', weights=[0.9, 1.2, 1.1], n_jobs=-1)  # 建立組合評估模型
cv = StratifiedKFold(8)  # 設置交叉驗證方法

start = time.time()
cv_score = cross_val_score(model_vot, X, y, cv=cv)  # 交叉驗證
end = time.time()
print(end - start)

print ('{:*^60}'.format('Cross val socres:'))
print (cv_score)  # 印出每次交叉驗證得分
print ('Mean scores is: %.2f' % cv_score.mean())  # 印出平均交叉驗證得分

start = time.time()
model_vot.fit(X, y)  # 模型訓練
end = time.time()
print(end - start)
#VotingClassifier(estimators=[('randomforest', RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#            max_depth=None, max_features='auto', max_leaf_nodes=None,
#            min_impurity_decrease=0.0, min_impurity_split=None,
#            min_samples_leaf=1, min_samples_split=2,
#n_estimators=20, n_jobs=1, oob_score=False, random_state=0,
#         verbose=0, warm_start=False))],
#         flatten_transform=None, n_jobs=-1, voting='soft',
#         weights=[0.9, 1.2, 1.1])

# 新數據集做預測
X_raw_data = pd.read_csv('./data/new_abnormal_orders.txt', dtype=dtypes)  # 讀取要預測的數據集
X_raw_new = X_raw_data.iloc[:, 1:]  # 分割輸入變量X，並丟棄訂單ID行和最後一行目標變量
str2int_data_new = str2int(X_raw_new, model_dvtransform, unique_object, training=False)  # 字符串分類轉整數型分類
datetime2int_data_new = datetime2int(X_raw_new)  # 日期時間轉換
combine_set_new = np.hstack((str2int_data_new, datetime2int_data_new))  # 合併轉換後的分類和拓展後的日期數據集
constant_set_new = X_raw_new[['total_money', 'total_quantity']]  # 原始連續數據變量
X_combine_new = np.hstack((combine_set_new, constant_set_new))  # 再次合併數據集
y_predict = model_vot.predict(X_combine_new)  # 預測結果
print ('{:*^60}'.format('Predicted Labesls:'))
print (y_predict)  # 印出預測值
# /Users/Vince/anaconda/lib/python3.6/site-packages/sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
#  if diff:
