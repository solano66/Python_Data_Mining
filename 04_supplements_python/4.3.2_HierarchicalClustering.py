'''
Collated by Ching-Shih Tsou 鄒慶士 博士 (Ph.D.) Distinguished Prof. at the Department of Mechanical Engineering/Director at the Center of Artificial Intelligence & Data Science (機械工程系特聘教授兼人工智慧暨資料科學研究中心主任), MCUT (明志科技大學); Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), NTUB (國立臺北商業大學)
Notes: This code is provided without warranty.
'''

import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch # scipy 1.5.0 -> 1.4.1 (No need to downgrade the version of scipy 2021/Nov./18)

# =============================================================================
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
# =============================================================================

#  讀入R 語言美國汽車雜誌道路測試資料
mtcars = pd.read_csv('mtcars.csv', encoding='utf-8', index_col=0)
mtcars.info()
mtcars.head()

# 產生觀測值間的距離值，設為歐幾里德距離
dist = sch.distance.pdist(mtcars, 'euclidean') # Pairwise distances between observations in n-dimensional space.
type(dist) # 長度496的ndarray (Why? (32*32-32)/2, same as R
dist.shape

# 根據距離進行聚合法階層式集群
# 群間距離計算方法設為最遠距離法 (complete)
hc = sch.linkage(dist, method='complete') 
# 0~30共聚合31次，後續31.....編號折返回來抓集群結果

# A (n-1) by 4 matrix Z is returned. At the i-th iteration, clusters with indices Z[i, 0] and Z[i, 1] are combined to form cluster n + i. A cluster with an index less than n corresponds to one of the original observations. The distance between clusters Z[i, 0] and Z[i, 1] is given by Z[i, 2]. The fourth value Z[i, 3] represents the number of original observations in the newly formed cluster.
hc

# 繪製樹狀圖( 圖4.8)    
plt.figure(figsize=(10, 7))
plt.title("Cluster Dendrogram")
dn = sch.dendrogram(hc, labels=mtcars.index.to_list(), leaf_rotation=90, leaf_font_size=14) # Just in case someone else is searching for the same issue, by converting the labels to list, it will work.
# Dendrogram: ValueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all() (https://stackoverflow.com/questions/65430814/dendrogram-valueerror-the-truth-value-of-a-series-is-ambiguous-use-a-empty-a)

# 第一橫列 0 與 1 表示第一次聚合編號 0 和 1 的樣本，第三個欄位表示前兩個群之間的距離，第四個欄位表示新生成聚類群所包含的元素的個數
tmp = hc
mtcars.index[0:2]

# 第四次聚合是編號 13 的樣本與前面第二群中編號 11 與 12 的樣本
mtcars.index[13]
mtcars.index[11:13]

# 每次聚合對象間的距離值，總共聚合31次
hc[:, -2]
len(hc)

#### Reference:
# How to measure clustering performances when there are no ground truth?
https://medium.com/@haataa/how-to-measure-clustering-performances-when-there-are-no-ground-truth-db027e9a871c
