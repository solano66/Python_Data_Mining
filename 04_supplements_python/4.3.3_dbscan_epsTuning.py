'''
Collated by Ching-Shih Tsou 鄒慶士 博士 (Ph.D.) Distinguished Prof. at the Department of Mechanical Engineering/Director at the Center of Artificial Intelligence & Data Science (機械工程系特聘教授兼人工智慧暨資料科學研究中心主任), MCUT (明志科技大學); Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), NTUB (國立臺北商業大學)
Notes: This code is provided without warranty.
'''

#### DBSCAN Python Example: The Optimal Value For Epsilon (EPS)
import numpy as np
# from sklearn.datasets.samples_generator import make_blobs # 舊版sklearn '0.23.2'用法，新版'0.24.1'以後不適用
from sklearn.datasets import make_blobs # Some simulation tools except default datasets

from sklearn.neighbors import NearestNeighbors # functions to find nearest neighbors 鄰域中近鄰計算函數
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
# 隨機產生四群二維樣本點 Randomly generate four groups of points in a two-dimenional space
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
import pandas as pd
pd.DataFrame(X).describe()


# 檢視y的可能值與次數分佈 Inspect the frequency of y
np.unique(y, return_counts=True) # 300/4 = 75 points for each cluster

# 二維樣本散佈圖 Make a scatterplot
plt.scatter(X[:,0], X[:,1])

#### DBSCAN (Density-Based Spatial Clustering with Applications to Noises)
# In comparison to other clustering algorithms, DBSCAN is particularly well suited for problems which require:

# 1. Minimal domain knowledge to determine the input parameters (i.e. K in k-means and Dmin in hierarchical clustering) 輸入參數較不需要領域知識
# 2. Discovery of clusters with arbitrary shapes 可發掘任意形狀的群
# 3. Good efficiency on large databases 大型資料庫效率佳

#### Algorithm
# As is the case in most machine learning algorithms, the model’s behaviour is dictated by several parameters. In the proceeding article, we’ll touch on three hyperparameters. 算法的三個超參數

# 1. eps: Two points are considered neighbors if the distance between the two points is below the threshold epsilon. 圓形鄰域半徑距離
# 2. min_samples: The minimum number of neighbors a given point should have in order to be classified as a core point. It’s important to note that the point itself is included in the minimum number of samples. 被歸為核心點的最少鄰居數(包括自已)
# 3. metric: The metric to use when calculating distance between instances/examples/samples in a feature array (i.e. euclidean distance). 距離計算方式

# The algorithm works by computing the distance between every point and all other points. We then place the points into one of three categories. 三種類型的點

# 1. Core point: A point with at least min_samples points whose distance with respect to the point is below the threshold defined by epsilon. 核心點，如上所述
# 2. Border point: A point that isn’t in close proximity to at least min_samples points but is **close enough to one or more core point**. Border points are included in the cluster of the closest core point. 邊界點，沒有最少鄰居數的鄰居樣本，但與一個或多個核心點夠接近
# 3. Noise point: Points that aren’t close enough to core points to be considered border points. Noise points are ignored. That is to say, they aren’t part of any cluster. 以上皆非(噪訊點)

#### Hyperparameter tuning
# The following paper, describes an approach for automatically determining the optimal value for Eps.
# https://iopscience.iop.org/article/10.1088/1755-1315/31/1/012012/pdf
# In layman’s terms, we find a suitable value for epsilon by calculating the distance to the nearest n points for each point, sorting and plotting the results. Then we look to see where the change is most pronounced/remarkable (think of the angle between your arm and forearm) and select that as epsilon. 計算每個點與其第n個鄰居點的距離，接著觀察何處變化最大！

neigh = NearestNeighbors(n_neighbors=2) # Define an empty model first (Why n_neighbors = 2 ? Exclude itself.)
type(neigh) # sklearn.neighbors._unsupervised.NearestNeighbors

nbrs = neigh.fit(X) # A parametrized model
type(nbrs) # sklearn.neighbors._unsupervised.NearestNeighbors same as above

[set(dir(nbrs)) - set(dir(neigh))] # An empty set [set()]

help(nbrs.kneighbors) # Finds the K-neighbors of a point. Returns indices of and distances to the neighbors of each point. So the values in the first column are always zeros.

distances, indices = nbrs.kneighbors(X)

distances # (300 points, 2 neighboring distances including itself)

indices # (300 points, 2 neighbors' indices including itself)

distances = np.sort(distances, axis=0) # (300 points, 2 neighboring distances in increasing order)

# Take the sorting distances (excluding itself) out and make a line plot 用排序好的近鄰距離，繪製折線圖
distances = distances[:,1]
plt.plot(distances)

#### Selecting 0.3 for eps (Why 0.3? Please check above plot 看圖) and setting min_samples to 5 to train our model.
m = DBSCAN(eps=0.3, min_samples=5)
m.fit(X)

clusters = m.labels_ # (300, )
# 檢視群編號的可能值與次數分佈 Inspect the clustering results (labels) and its frequency distribution
np.unique(clusters, return_counts=True) # Totally we have 11 + 1 clusters


colors = ['royalblue', 'maroon', 'forestgreen', 'mediumorchid', 'tan', 'deeppink', 'olive', 'goldenrod', 'lightcyan', 'navy', 'red', 'green']
len(colors) # 12 colors
vectorizer = np.vectorize(lambda x: colors[x % len(colors)]) # % modulo operator

plt.scatter(X[:,0], X[:,1], c=vectorizer(clusters))
#plt.legend()
#plt.show()

#### Selecting 0.2 for eps and setting min_samples to 5 to train our model.
m = DBSCAN(eps=0.2, min_samples=5)
m.fit(X)

clusters = m.labels_ # (300, )
# 檢視群編號的可能值與次數分佈 Inspect the clustering results (labels) and its frequency distribution
np.unique(clusters, return_counts=True) # Totally we have 9 + 1 clusters


colors = ['royalblue', 'maroon', 'forestgreen', 'mediumorchid', 'tan', 'deeppink', 'olive', 'goldenrod', 'lightcyan', 'navy']
len(colors) # 10 colors
vectorizer = np.vectorize(lambda x: colors[x % len(colors)]) # % modulo operator

plt.scatter(X[:,0], X[:,1], c=vectorizer(clusters))
#plt.legend()
#plt.show()

#### *** Selecting 0.6 for eps and setting min_samples to 5 to train our model.
m = DBSCAN(eps=0.6, min_samples=5)
m.fit(X)

clusters = m.labels_ # (300, )
# 檢視群編號的可能值與次數分佈 Inspect the clustering results (labels) and its frequency distribution
np.unique(clusters, return_counts=True) # Totally we have 4 + 1 clusters


colors = ['royalblue', 'maroon', 'forestgreen', 'mediumorchid', 'tan']
len(colors) # 5 colors
vectorizer = np.vectorize(lambda x: colors[x % len(colors)]) # % modulo operator

plt.scatter(X[:,0], X[:,1], c=vectorizer(clusters))
#plt.legend()
#plt.show()

#### *** Selecting 0.6 for eps and setting min_samples to 2 to train our model. 降低最小樣本數後，離群點減少！
m = DBSCAN(eps=0.6, min_samples=2)
m.fit(X)

clusters = m.labels_ # (300, )
# 檢視群編號的可能值與次數分佈 Inspect the clustering results (labels) and its frequency distribution
np.unique(clusters, return_counts=True) # Totally we have 4 + 1 clusters


colors = ['royalblue', 'maroon', 'forestgreen', 'mediumorchid', 'tan']
len(colors) # 5 colors
vectorizer = np.vectorize(lambda x: colors[x % len(colors)]) # % modulo operator

plt.scatter(X[:,0], X[:,1], c=vectorizer(clusters))
#plt.legend()
#plt.show()

#### Final Thoughts
# Unlike k-means, DBSCAN will figure out the number of clusters. DBSCAN works by determining whether the minimum number of points are close enough to one another to be considered part of a single cluster. 算法DBSCAN會自動思索出群數 DBSCAN is very sensitive to scale since epsilon is a fixed value for the maximum distance between two points. (send data to DBSCAN after scaling) 對資料尺度相當敏感

### References
# https://towardsdatascience.com/machine-learning-clustering-dbscan-determine-the-optimal-value-for-epsilon-eps-python-example-3100091cfbc
# https://geodacenter.github.io/workbook/99_density/lab9b.html



