'''
Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the IDS, CICD, and CISD of NTUB (臺北商業大學資訊與決策科學研究所暨智能控制與決策研究室教授兼校務永續發展中心主任); at the ME Dept. and CAIDS of MCUT (2020年借調至明志科技大學機械工程系擔任特聘教授兼人工智慧暨資料科學研究中心主任兩年); the CSQ (2019年起任中華民國品質學會大數據品質應用委員會主任委員); the DSBA (2013年創立臺灣資料科學與商業應用協會); and the CARS (2012年創立中華R軟體學會) 
Notes: This code is provided without warranty.
'''

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Reading the Iris.csv file
data = load_iris()

# Extracting Attributes / Features
X = data.data

# Extracting Target / Class Labels
y = data.target

# 載入sklearn 套件的樹狀模型模組tree
from sklearn import tree
# 宣告DecisionTreeClassifier() 類別空模clf(未更改預設設定)
clf = tree.DecisionTreeClassifier()
# 傳入訓練資料擬合實模clf
clf = clf.fit(X, y)
# ValueError: could not convert string to float: '> 200 DM' (前面如果字串自變數未標籤編碼，則會報錯！)
# 預測訓練集標籤train_pred
pred = clf.predict(X)
print(' 錯誤率為{0}.'.format(np.mean(y != pred)))

print('此樹有{}節點'.format(clf.tree_.node_count)) # 好複雜的一棵樹！所以接下來我們調參
# 可以先繪製樹狀圖來看看(先跳至Line 47)

print(clf.get_params())
keys = ['max_depth', 'max_leaf_nodes', 'min_samples_leaf']
# type(clf.get_params()) # dict
print([(key, clf.get_params().get(key)) for key in keys])

# 再次宣告空模clf(更改上述三參數設定)、配適與預測
clf = tree.DecisionTreeClassifier(max_leaf_nodes = 10,
min_samples_leaf = 7, max_depth= 2) # max_depth: 30 -> 3 -> 2
clf = clf.fit(X, y)
pred = clf.predict(X)
print(' 錯誤率為{0}.'.format(np.mean(y != pred)))


# 載入Python 語言字串讀寫套件
from io import StringIO
# import pydot
import pydotplus # conda install pydotplus --y (會**自動**安裝2.38版本的graphviz，另外pandas需1.1.3以前的版本)
# 將樹tree 輸出為StringIO 套件的dot_data
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, feature_names=
list(data.feature_names), class_names = data.target_names,
filled=True, rounded=True) # filled=True, rounded=True

# An alternative way on Windows
# import os
# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/' # 裝到C:/Program Files (x86)/Graphviz2.38/

# dot_data 轉為graph 物件
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph 寫出pdf
graph.write_pdf("iris4.pdf")


