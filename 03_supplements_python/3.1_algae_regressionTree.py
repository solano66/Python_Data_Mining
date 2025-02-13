'''
Collated by Ching-Shih Tsou 鄒慶士 博士 (Ph.D.) Distinguished Prof. at the Department of Mechanical Engineering/Director at the Center of Artificial Intelligence & Data Science (機械工程系特聘教授兼人工智慧暨資料科學研究中心主任), MCUT (明志科技大學); Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), NTUB (國立臺北商業大學)
Notes: This code is provided without warranty.
'''

import pandas as pd
algae = pd.read_csv('algae.csv') # 此處不可將字串變量設定為Category/Factor型別！

# 將文字資料轉為數值(編碼)
algae['season'] = algae['season'].map({'spring':1,'summer':2,'autumn':3,'winter':4}).astype(int) # label encoding
algae['size'] = algae['size'].map({'small':1,'medium':2,'large':3}).astype(int)
algae['speed'] = algae['speed'].map({'low':1,'medium':2,'high':3}).astype(int)

algae.info()

# 移除遺缺程度嚴重的樣本(Python的樹狀模型實現不理想，此處仍須對遺缺樣本進行處理，i.e. 移除或填補！)
cleanAlgae = algae.dropna(axis='rows', thresh=13)

# 以各變項中位數填補遺缺值（不填補scikit-learn.tree會報錯！顯然scikit-learn實現得不好，R語言無需填補即可建樹）
# cleanAlgae = pd.DataFrame()
for col in algae.columns:
    cleanAlgae[col] = algae[col].fillna(algae[col].median())

#### Data splitting by holdout method 保留法資料切分
from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(algae.iloc[:,:11], algae['a1'], test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(cleanAlgae.iloc[:,:11], cleanAlgae['a1'], test_size=0.2, random_state=42)

# X_train, X_test, y_train, y_test = train_test_split(algae.iloc[:,:11], algae['a1'], test_size=0.2, random_state=42)
#### 配適迴歸樹 Fit a regression tree
# Step 1
from sklearn.tree import DecisionTreeRegressor # DecisionTreeClassifier (for categorical y)

# Step 2
# Missing values in scikits machine learning (https://stackoverflow.com/questions/9365982/missing-values-in-scikits-machine-learning)
reg1 = DecisionTreeRegressor() # 使用預設參數試建模
# Step 3
reg1.fit(X_train, y_train) # 前三個變數不得為'Category'
# ValueError: Input contains NaN, infinity or a value too large for dtype('float32').
# ValueError: could not convert string to float: 'winter' on 2022/Sep./16

dir(reg1)
dir(reg1.tree_)
# 節點數271個，過度茂盛，顯示可能過度配適(過擬合)狀況嚴重
n_nodes = reg1.tree_.node_count
print(' 迴歸樹有{0} 節點。'.format(n_nodes))

y_train_pred = reg1.predict(X_train)
y_test_pred = reg1.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score # MSE, RSME, R2

# 訓練集與測試集的MSE或RMSE差距非常大！(請參見圖3.3)
mean_squared_error(y_train, y_train_pred) # 居然是 0 ! 模型與訓練樣本配適的如此完美！
mean_squared_error(y_test, y_test_pred) # > 495
# RMSE
import numpy as np
np.sqrt(mean_squared_error(y_train, y_train_pred))
np.sqrt(mean_squared_error(y_test, y_test_pred))

r2_score(y_train, y_train_pred) # 居然是 1 !
r2_score(y_test, y_test_pred) # 負的判定係數(When is R squared negative? https://stats.stackexchange.com/questions/12900/when-is-r-squared-negative)

#### Tree model report
children_left = reg1.tree_.children_left
s1 = ' 各節點的左子節點分別是{0}'
s2 = '\n{1}(-1 表葉子節點沒有子節點)。'
print(''.join([s1, s2]).format(children_left[:9],
children_left[9:]))

children_right = reg1.tree_.children_right
s1 = ' 各節點的右子節點分別是{0}'
s2 = '\n{1}(-1 表葉子節點沒有子節點)。'
print(''.join([s1, s2]).format(children_right[:9],
children_right[9:]))

feature = reg1.tree_.feature
s1 = ' 各節點分支屬性索引為(-2 表無分支屬性)'
s2 = '\n{0}。'
print(''.join([s1, s2]).format(feature))

threshold = reg1.tree_.threshold
s1 = ' 各節點分支屬性門檻值為(-2 表無分支屬性門檻值)'
s2 = '\n{0}\n{1}\n{2}\n{3}。'
print(''.join([s1, s2]).format(threshold[:6],
threshold[6:12], threshold[12:18], threshold[18:]))

# 各節點樹深串列node_depth
node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
# 各節點是否為葉節點的真假值串列
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
# 值組(節點編號, 父節點深度) 形成的堆疊串列, 初始化時只有根節點
stack = [(0, -1)]
# 從堆疊逐一取出資訊產生報表，堆疊最終會變空
while len(stack) > 0:
    node_i, parent_depth = stack.pop()
    # 自己的深度為父節點深度加1
    node_depth[node_i] = parent_depth + 1
    # 如果是測試節點(i.e. 左子節點不等於右子節點)，而非葉節點
    if (children_left[node_i] != children_right[node_i]):
    # 加左分枝節點，分枝節點的父節點深度正是自己的深度
        stack.append((children_left[node_i],parent_depth+1))
    # 加右分枝節點，分枝節點的父節點深度正是自己的深度
        stack.append((children_right[node_i],parent_depth+1))
    else:
    # is_leaves 原預設全為False，最後有True 有False
        is_leaves[node_i] = True

print(" 各節點的深度分別為：{0}".format(node_depth))

print(" 各節點是否為終端節點的真假值分別為：\n{0}\n{1}"
.format(is_leaves[:10], is_leaves[10:]))

print("%s 個節點的二元樹結構如下：" % n_nodes)
# 迴圈控制敘述逐一印出分類樹模型報表

for i in range(n_nodes):
    if is_leaves[i]:
        print("%snd=%s leaf nd."%(node_depth[i]*" ", i))
    else:
        s1 = "%snd=%s test nd: go to nd %s"
        s2 = " if X[:, %s] <= %s else to nd %s."
        print(''.join([s1, s2])
        % (node_depth[i] * " ",
        i,
        children_left[i],
        feature[i],
        threshold[i],
        children_right[i],
        ))

print()

#### Tree plotting
# try:
#     from StringIO import StringIO
# except ImportError:
#     from io import StringIO

from io import StringIO
import pydotplus # !conda install pydotplus --y (會自動安裝2.38版本的graphviz，*另外pandas需1.1.3以前的版本*, 2022/Sep./22 Python 3.9 + pandas 1.4.2 okay) or pip install pydotplus
# conda install -c anaconda graphviz --y
# What is StringIO in python used for in reality? https://stackoverflow.com/questions/7996479/what-is-stringio-in-python-used-for-in-reality

dot_data = StringIO() # 樹狀模型先output到dot_data
from sklearn import tree
tree.export_graphviz(reg1, out_file=dot_data, feature_names=['season', 'size', 'speed', 'mxPH', 'mnO2', 'Cl', 'NO3', 'NH4', 'oPO4', 'PO4', 'Chla']) 
#(graph,) = pydot.graph_from_dot_data(dot_data.getvalue())

graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) # 從dot_data產生graph
graph.write_pdf("algae_rt1.pdf") # Terrible ! A very complex tree.

#### 援引R語言rpart的默認超參數設定值
# 再次宣告空模reg2(更改為R 語言套件{rpart} 的預設值，1976年前身S語言至今46年經驗，其建模超參數預設值通常較佳！反觀scikit-learn 2013八月)
reg2 = DecisionTreeRegressor(max_leaf_nodes = 10, min_samples_leaf = 7, max_depth= 30) # 一株樹至多10個葉節點、落入葉節點的最小樣本數為7、樹深最大為30層(每次分支最小樣本數minsplit = 20, 葉節點最小樣本數 minbucket = round(minsplit/3), cp = 0.01,               maxcompete = 4, maxsurrogate = 5, usesurrogate = 2, xval = 10, surrogatestyle = 0, maxdepth = 30)
reg2.fit(X_train, y_train)

# 節點數19 個，顯示配適結果改善
n_nodes = reg2.tree_.node_count
print(' 迴歸樹有{0} 節點。'.format(n_nodes))

# 訓練樣本績效評估
y_train_pred = reg2.predict(X_train)

# MSE
mean_squared_error(y_train, y_train_pred)

# RMSE
import numpy as np
np.sqrt(mean_squared_error(y_train, y_train_pred))

# R^2判定係數
r2_score(y_train, y_train_pred)


# 測試樣本績效評估
y_test_pred = reg2.predict(X_test)

# MSE
mean_squared_error(y_test, y_test_pred)

# RMSE
import numpy as np
np.sqrt(mean_squared_error(y_test, y_test_pred))

# R^2判定係數
r2_score(y_test, y_test_pred)

# 多小？或差距為何？方能停止訓練！
# 端視y的分佈，以及其管制界線

#### Tree model report again
children_left = reg2.tree_.children_left
s1 = ' 各節點的左子節點分別是{0}'
s2 = '\n{1}(-1 表葉子節點沒有子節點)。'
print(''.join([s1, s2]).format(children_left[:9],
children_left[9:]))

children_right = reg2.tree_.children_right
s1 = ' 各節點的右子節點分別是{0}'
s2 = '\n{1}(-1 表葉子節點沒有子節點)。'
print(''.join([s1, s2]).format(children_right[:9],
children_right[9:]))

feature = reg2.tree_.feature
s1 = ' 各節點分支屬性索引為(-2 表無分支屬性)'
s2 = '\n{0}。'
print(''.join([s1, s2]).format(feature))

threshold = reg2.tree_.threshold
s1 = ' 各節點分支屬性門檻值為(-2 表無分支屬性門檻值)'
s2 = '\n{0}\n{1}\n{2}\n{3}。'
print(''.join([s1, s2]).format(threshold[:6],
threshold[6:12], threshold[12:18], threshold[18:]))

# 各節點樹深串列node_depth
node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
# 各節點是否為葉節點的真假值串列
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
# 值組(節點編號, 父節點深度) 形成的堆疊串列, 初始化時只有根節點
stack = [(0, -1)]
# 從堆疊逐一取出資訊產生報表，堆疊最終會變空
while len(stack) > 0:
    node_i, parent_depth = stack.pop()
    # 自己的深度為父節點深度加1
    node_depth[node_i] = parent_depth + 1
    # 如果是測試節點(i.e. 左子節點不等於右子節點)，而非葉節點
    if (children_left[node_i] != children_right[node_i]):
    # 加左分枝節點，分枝節點的父節點深度正是自己的深度
        stack.append((children_left[node_i],parent_depth+1))
    # 加右分枝節點，分枝節點的父節點深度正是自己的深度
        stack.append((children_right[node_i],parent_depth+1))
    else:
    # is_leaves 原預設全為False，最後有True 有False
        is_leaves[node_i] = True

print(" 各節點的深度分別為：{0}".format(node_depth))

print(" 各節點是否為終端節點的真假值分別為：\n{0}\n{1}"
.format(is_leaves[:10], is_leaves[10:]))

print("%s 個節點的二元樹結構如下：" % n_nodes)
# 迴圈控制敘述逐一印出分類樹模型報表

for i in range(n_nodes):
    if is_leaves[i]:
        print("%snd=%s leaf nd."%(node_depth[i]*" ", i))
    else:
        s1 = "%snd=%s test nd: go to nd %s"
        s2 = " if X[:, %s] <= %s else to nd %s."
        print(''.join([s1, s2])
        % (node_depth[i] * " ",
        i,
        children_left[i],
        feature[i],
        threshold[i],
        children_right[i],
        ))

print()


#### Tree plotting again
from io import StringIO
import pydotplus # !conda install pydotplus --y (會自動安裝2.38版本的graphviz，*另外pandas需1.1.3以前的版本*, 2022/Sep./22 Python 3.9 + pandas 1.4.2 okay) or pip install pydotplus
# conda install -c anaconda graphviz --y
# What is StringIO in python used for in reality? https://stackoverflow.com/questions/7996479/what-is-stringio-in-python-used-for-in-reality

dot_data = StringIO() # 樹狀模型先output到dot_data
from sklearn import tree
tree.export_graphviz(reg2, out_file=dot_data, feature_names=['season', 'size', 'speed', 'mxPH', 'mnO2', 'Cl', 'NO3', 'NH4', 'oPO4','PO4', 'Chla']) 
#(graph,) = pydot.graph_from_dot_data(dot_data.getvalue())

graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) # 從dot_data產生graph
graph.write_pdf("algae_rt2.pdf")

#### Data splitting by cross-validation 交叉驗證 cross_val_score
from sklearn.model_selection import cross_val_score

help(cross_val_score)

from sklearn.tree import DecisionTreeRegressor

reg3 = DecisionTreeRegressor()
# 傳入模型、完整資料集(X and y)、摺數的設定
scores = cross_val_score(reg3, cleanAlgae.iloc[:,:11], cleanAlgae['a1'], cv=10)

scores.shape # (10, 0)

#### Cross-validation: evaluating estimator performance 交叉驗證 cross_validate
# https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation

from sklearn.model_selection import cross_validate
# from sklearn.metrics import mean_squared_error, r2_score
import sklearn
sorted(sklearn.metrics.SCORERS.keys())

scoring = ['neg_mean_squared_error', 'neg_median_absolute_error']

scores = cross_validate(reg3, cleanAlgae.iloc[:,:11], cleanAlgae['a1'], scoring=scoring)
sorted(scores.keys())

scores['test_neg_mean_squared_error']









