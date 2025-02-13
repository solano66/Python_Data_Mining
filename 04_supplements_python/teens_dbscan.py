
### 4.3.1.1 青少年市場區隔案例 by DBSCAN
## ------------------------------------------------------------------------
import numpy as np
import pandas as pd
teens = pd.read_csv("./_data/snsdata.csv")
# 文件詞項矩陣前面加上入學年、性別、年齡與朋友數等欄位
print(teens.shape) # 30000 * (4 + 36 terms by NLP)

# 留意gradyear 的資料型別
print(teens.dtypes)

# gradyear 更新為字串str 型別
teens['gradyear'] = teens['gradyear'].astype('str')
# 除了資料型別外，ftypes 還報導了屬性向量是稀疏還是稠密的
# print(teens.ftypes.head()) # deprecated after 1.0.0

# 各變數敘述統計值(報表過寬，只呈現部份結果)
print(teens.describe(include='all'))

# 各欄位遺缺值統計(只有gender 與age 有遺缺)(注意！此處並未進行遺缺值處理)
print(teens.isnull().sum().head())

# 各詞頻變數標準化建模
from sklearn.preprocessing import StandardScaler

sc = StandardScaler() # Object-oriented programming paradigm

# 配適與轉換接續完成函數
teens_z = sc.fit_transform(teens.iloc[:,4:])

# 錯誤用法！Python的類別函數不可使用泛函式編程語法
# teens_z = StandardScaler(teens.iloc[:,4:]) # > scikit-learn 0.23.2 可以！

# scikit-learn下preprocessing模組的scale()函數可用泛函式編程語法
from sklearn.preprocessing import scale
teens_z = scale(teens.iloc[:,4:])

# 資料導向程式設計經常輸出與輸入不同調(DataFrame 入ndarray 出)
print(type(teens_z))

# 轉為資料框物件取用describe() 方法確認標準化結果
print(pd.DataFrame(teens_z[:,30:33]).describe())
# mean 5.494864e-17  1.136868e-17 -9.687066e-17 可能數字上會有差異，這說明二進位制的計算機的數值運算不穩定性(numerical instability)

# Python DBSCAN集群
from sklearn.cluster import DBSCAN
mdl = DBSCAN(eps=5, min_samples=5) # More computation time than k-means and for larger 'eps' and 'min_samples'.
# eps=3.8, min_samples=10

# 配適前空模的屬性與方法
pre = dir(mdl)
# 空模的幾個屬性與方法
print(pre[31:36])
# 以標準化文件詞項矩陣配適集群模型
import time
start = time.time()
mdl.fit(teens_z)
end = time.time()
print("DBSCAN fitting spent {} seconds".format(end - start))

# 配適後實模的屬性與方法
post = dir(mdl)
# 實模的幾個屬性與方法
print(post[51:56])

# 實模與空模屬性和方法的差異(前或後有下底線_)
print(list(set(post) - set(pre)))

# res.labels_ 為30,000 名訓練樣本的歸群標籤
# import sys
# np.set_printoptions(threshold=sys.maxsize)
print(mdl.labels_.shape)

# 各群人數分佈(思考numpy下如何做！) 317群
print(pd.Series(mdl.labels_).value_counts())

# 核心樣本點的編號
dir(mdl)
mdl.core_sample_indices_
len(mdl.core_sample_indices_) # 26845

# 前10 個樣本的群編號
print (mdl.labels_[:10])

# 26845核心樣本點的特徵數值
print(mdl.components_.shape)
print(mdl.components_)

##### 以下請同學練習如何修改(2024/4/11詢問結果！)
# 轉換成pandas 資料框，給予群編號與字詞名稱，方便結果詮釋
cen = pd.DataFrame(res.cluster_centers_, index = range(5),
columns = teens.iloc[:,4:].columns)
print(cen)

# 每次歸群結果的釋義會有不同
# Princesses: 1
# Criminals: 4
# Basket Cases: 0
# Athletes: 3
# Brains: 2

# 各群中心座標矩陣轉置後繪圖
ax = cen.T.plot() # seaborn, ggplot or pandas ?
# 低階繪圖設定x 軸刻度位置
ax.set_xticks(list(range(36)))
# 低階繪圖設定x 軸刻度說明文字
ax.set_xticklabels(list(cen.T.index), rotation=90)
fig = ax.get_figure()
fig.tight_layout()
# fig.savefig('./_img/sns_lineplot.png')

# 以下為課本/講義沒有的補充程式碼，主要在進行事後(建模後)的分析
# 添加群編號於原資料表後
teens = pd.concat([teens, pd.Series(mdl.labels_).rename('cluster')], axis=1)

# 抓集群未使用的三個變量(剛才歸群時未用，但事後分析確有助於了解各群的異同，以及歸群結果的品質)
teens[['gender','age','friends','cluster']][0:5]

# 各群平均年齡(群組與摘要也！)
teens.groupby('cluster').aggregate({'age': np.mean}) # 同儕間年齡差異不大！

# 新增是否為女生欄位'female'
teens.gender.value_counts()
teens.gender.value_counts(dropna = False)

# Equivalent of R/ifelse in Python/Pandas? Compare string columns? (https://stackoverflow.com/questions/35666272/equivalent-of-r-ifelse-in-python-pandas-compare-string-columns#)
def if_this_else_that(x, list_of_checks, yes_label, no_label):
    if x in list_of_checks:
        res = yes_label
    else: 
        res = no_label
    return(res)

teens['female'] = teens['gender'].apply(lambda x: if_this_else_that(x, ['F'], True, False))

teens[['gender', 'female']].head(n=20)

teens.female.sum() # 22054

# 各群女生人數比例(群組與摘要也！)
teens.groupby('cluster').aggregate({'female': np.mean})

# 各群朋友數(群組與摘要也！)
teens.groupby('cluster').aggregate({'friends': np.mean})