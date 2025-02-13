'''
Collated by Ching-Shih Tsou 鄒慶士 博士 (Ph.D.) Distinguished Prof. at the Department of Mechanical Engineering/Director at the Center of Artificial Intelligence & Data Science (機械工程系特聘教授兼人工智慧暨資料科學研究中心主任), MCUT (明志科技大學); Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), NTUB (國立臺北商業大學)
Notes: This code is provided without warranty.
'''

### 4.3.1.1 青少年市場區隔案例 Teenagers market segementation case
## ------------------------------------------------------------------------
import numpy as np
import pandas as pd
teens = pd.read_csv("./_data/snsdata.csv")
# 文件詞項矩陣前面加上入學年、性別、年齡與朋友數等欄位
print(teens.shape) # 30000 * (4 + 36)

# 留意gradyear 的資料型別
print(teens.dtypes)

# gradyear 更新為字串str 型別
teens['gradyear'] = teens['gradyear'].astype('str')
# 除了資料型別外，ftypes 還報導了屬性向量是稀疏還是稠密的
# print(teens.ftypes.head())

# 各變數敘述統計值(報表過寬，只呈現部份結果)
print(teens.describe(include='all'))

# 各欄位遺缺值統計(只有gender 與age 有遺缺)
print(teens.isnull().sum().head())

# 各詞頻變數標準化建模
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# 配適與轉換接續完成函數

teens_z = sc.fit_transform(teens.iloc[:,4:])

# 資料導向程式設計經常輸出與輸入不同調(DataFrame 入ndarray 出)
print(type(teens_z))

# 轉為資料框物件取用describe() 方法確認標準化結果
print(pd.DataFrame(teens_z[:,30:33]).describe())

# Python k 平均數集群，隨機初始化的集群結果通常比較好
from sklearn.cluster import KMeans
mdl = KMeans(n_clusters=5, init='random')

# 配適前空模的屬性與方法
pre = dir(mdl)
# 空模的幾個屬性與方法
print(pre[51:56])

# 以標準化文件詞項矩陣配適集群模型
mdl.fit(teens_z)
# 配適後實模的屬性與方法
post = dir(mdl)
# 實模的幾個屬性與方法
print(post[51:56])

# 實模與空模屬性和方法的差異
print(list(set(post) - set(pre)))

pd.Series(mdl.labels_).value_counts()

#### Use silhouette score ([-1, 1]) to evaluate 2 to 5 for number of clusters 側影分數評估分兩群到五群，何者較佳？ ####
## -----------------------------------
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
range_n_clusters = list(range(2,11))
print ("Number of clusters from 2 to 10: \n", range_n_clusters)

all_scores = []
avg_scores = []
for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters)
    # Use the fit_predict() to get the clustering results for each sample. 不光有fit_transform()，也有fit_predict()，取得樣本歸群結果
    preds = clusterer.fit_predict(teens_z)
    # Get the cluster centers matrix 抓群中心座標
    centers = clusterer.cluster_centers_ # (n_clusters, 36)
    
    # Calculate the silhouette score for each cluster 除入標準化數據矩陣與樣本歸群結果(方知同群與鄰群的樣本點)，計算各集群數下的平均側影分數
    score = silhouette_score(teens_z, preds, metric='euclidean')
    # Storing the silhouette scores 結果儲存
    avg_scores.append(score)
    print ("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))
    
    # Calculate the silhouette score for each sample 各樣本側影值分數
    score = silhouette_samples(teens_z, preds, metric='euclidean')
    # Storing the silhouette scores 結果儲存
    all_scores.append(score)

# *Subjective* or objective determination ? Which is better ? It depends. 主觀認定的集群個數(John Hughes)，及客觀科學的集群認定，哪一個比較好？ Context, context, and context.

# 檢視評估結果
import sys
np.set_printoptions(threshold = sys.maxsize)
# 30000青少年四種歸群結果各自的側影分數
all_scores
type(all_scores) # a list object
type(all_scores[2]) # numpy.ndarray

all_scores[2].shape # (30000,)
all_scores[2][:10] # k = 4 下前十個樣本的側影值分數

# 各集群數下所有樣本的平均側影分數
avg_scores

import matplotlib.pyplot as plt
plt.plot(avg_scores)
