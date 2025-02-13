import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

y = np.random.normal(0, 0.1, 500)
y.shape
type(y)
y = y.reshape(100, 5)
y.shape

n = 0
col = []
for i in range(5):
    n+=1
    a = f"t{n}"
    col.append(a)

col


def name_gen(name, num):
    n = 0
    col = []
    for i in range(num):
        n+=1
        a = f"{name}{n}"
        col.append(a)
    return col


col = name_gen('t', 5)

row = name_gen('g', 100)

y = pd.DataFrame(y)
y = y.set_axis(col, axis=1)
y = y.set_axis(row, axis=0)
y

sns.clustermap(y, cmap='viridis')
plt.show()


# ## Stepwise Approach with Tree Cutting

# https://seaborn.pydata.org/generated/seaborn.clustermap.html
# https://www.python-graph-gallery.com/404-dendrogram-with-heat-map
# https://stackoverflow.com/questions/48173798/additional-row-colors-in-seaborn-cluster-map
# https://seaborn.pydata.org/generated/seaborn.hls_palette.html
# https://www.python-graph-gallery.com/405-dendrogram-with-heatmap-and-coloured-leaves

matrix = pd.DataFrame(np.random.random_integers(0,1, size=(50,4)))
labels = np.random.random_integers(0,5, size=50)

lut = dict(zip(set(labels), sns.hls_palette(len(set(labels)), l=0.5, s=0.8)))
row_colors = pd.DataFrame(labels)[0].map(lut)

g=sns.clustermap(matrix, col_cluster=False, linewidths=0.1, cmap='coolwarm', row_colors=row_colors)
plt.show()

# Libraries
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
 
# Data set
df = pd.read_csv('mtcars.csv')

# Prepare a vector of color mapped to the 'cyl' column
my_palette = dict(zip(df.cyl.unique(), ["orange","yellow","brown"]))
row_colors = df.cyl.map(my_palette)
 
# plot
sns.clustermap(df, metric="correlation", method="single", cmap="Blues", standard_scale=1, row_colors=row_colors)
plt.show()

my_palette


# ## K-means Clustering

from sklearn.cluster import KMeans

mu_tr = y.mean(axis=0)
std_tr = y.std(axis=0)

y_scale = np.array(((y - mu_tr)/std_tr))
y_scale = y_scale.T
y_scale.shape

kmeans = KMeans(n_clusters=3, random_state=0).fit(y_scale.T)
kmeans.labels_

km = pd.DataFrame()
km[0] = row
km[1] = kmeans.labels_
km.T


# ## Fuzzy C-Means Clustering
# conda install -c conda-forge scikit-fuzzy
# pip install -U scikit-fuzzy
import skfuzzy as fuzz

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    y, # the data
    4, # desired number of cluster 
    2, # Array exponentiation applied to the membership function u_old at each iteration, where U_new = u_old ** m.
    error=0.005, # stopping criterion, stop early if the norm of (u[p] - u[p-1]) < error.
    maxiter=1000, # maximum number of itteration
    # maxiter=1000,
    init=None) # Initial fuzzy c-partitioned matrix.
   
# cluster center
# cntr

# Final fuzzy c-partitioned matrix.
# u

# Initial guess at fuzzy c-partitioned matrix
# u0

# Final Euclidian distance matrix.
# d

# Objective function history.
# jm

# Number of iterations run.
# p

# Final fuzzy partition coefficient.
# fpc

cntr.T.shape
cnt_df = pd.DataFrame(cntr.T).set_axis(row, axis=0)
cnt_df = round(cnt_df, 2)
cnt_df

cluster_membership = np.argmax(cntr, axis=0) 
cluster_membership.shape

fanny = pd.DataFrame()
fanny[0] = row
fanny[1] = cluster_membership 
fanny.T

fannyyMA = cnt_df > 0.20 # where this 0.20 came from?????

# fannyyMA


# referece 
# [here](https://scikit-fuzzy.readthedocs.io/en/latest/api/skfuzzy.html#cmeans)

# ## Principal Component Analysis Clustring
# scale the data first
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(y)
scaler = scaler.transform(y)
dtsc = scaler

from sklearn.decomposition import PCA

pca = PCA()
pca.fit(dtsc)

pct = pca.fit_transform(dtsc)
# to getting the principal component or PCs

pc1 = pct[:, 0]
pc2 = pct[:, 1]

plt.scatter(pc1, pc2)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

pc_result = pd.DataFrame()
pc_result[0] = row
pc_result[1] = pc1
pc_result[2] = pc2
pc_result.T
pc_result.head()
pc_result[2][1]



plt.figure(figsize=(8,5))
plt.scatter(pc_result[1], pc_result[2])

for i in range(pc_result.shape[0]):
    plt.text(x=pc_result[1][i],# +0.3
             y=pc_result[2][i],# +0.3
             s=pc_result[0][i], 
             fontdict=dict(color='red',size=10),
             bbox=dict(facecolor='yellow',alpha=0.5))
    
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()


# reference 
# [here](https://github.com/hashABCD/Publications/blob/main/Medium/How%20to%20Add%20Text%20Labels%20to%20Scatterplot.ipynb)
# [here](https://towardsdatascience.com/how-to-add-text-labels-to-scatterplot-in-matplotlib-seaborn-ec5df6afed7a)


# ## Multidimensional Scaling (MDS)

loc = pd.read_csv("loc.csv")
index_name = ['Athens','Barcelona', 'Brussels', 'Clais', 'Cherbourg',
              'Clogne', 'Copenhagen', 'Geneva', 'Gibraltar', 'Hamburg',
              'Hook of Holland', 'Lisbon', 'Lyons', 'Madrid', 'Marseilles',
              'Milan', 'Munich', 'Paris', 'Rome', 'Stockholm', 'Vienna']

loc['city'] = index_name
loc.head()

plt.scatter(loc.V1, loc.V2)
plt.show()


plt.figure(figsize=(8,5))
plt.scatter(loc.V1, loc.V2)

for i in range(loc.shape[0]):
    plt.text(x=loc.V1[i],# +0.3
             y=loc.V2[i],# +0.3
             s=loc.city[i], 
             fontdict=dict(color='red',size=10),
             bbox=dict(facecolor='yellow',alpha=0.5))
plt.title('cmdscale(eurodist)')
plt.show()


# ## Biclustering
from sklearn.datasets import make_checkerboard
from sklearn.cluster import SpectralBiclustering
from sklearn.metrics import consensus_score


n_clusters = (4, 3)
data, rows, columns = make_checkerboard(
    shape=(300, 300), n_clusters=n_clusters, noise=10, shuffle=False, random_state=0
)

plt.matshow(data, cmap=plt.cm.Blues)
plt.title("Original dataset")
plt.show()


# shuffle clusters
rng = np.random.RandomState(0)
row_idx = rng.permutation(data.shape[0])
col_idx = rng.permutation(data.shape[1])
data = data[row_idx][:, col_idx]

plt.matshow(data, cmap=plt.cm.Blues)
plt.title("Shuffled dataset")
plt.show()


model = SpectralBiclustering(n_clusters=n_clusters, method="log", random_state=0)
model.fit(data)
score = consensus_score(model.biclusters_, (rows[:, row_idx], columns[:, col_idx]))

print("consensus score: {:.1f}".format(score))

fit_data = data[np.argsort(model.row_labels_)]
fit_data = fit_data[:, np.argsort(model.column_labels_)]

plt.matshow(fit_data, cmap=plt.cm.Blues)
plt.title("After biclustering; rearranged to show biclusters")
plt.show()


plt.matshow(
    np.outer(np.sort(model.row_labels_) + 1, np.sort(model.column_labels_) + 1),
    cmap=plt.cm.Blues,
)
plt.title("Checkerboard structure of rearranged data")
plt.show()


# reference 
# [here](https://scikit-learn.org/stable/auto_examples/bicluster/plot_spectral_biclustering.html)


# ### Calculate jaccard similarity in Python
A = {1, 2, 3, 5, 7}
B = {1, 2, 4, 8, 9}


def jaccard_similarity(A, B):
    #Find intersection of two sets
    nominator = A.intersection(B)

    #Find union of two sets
    denominator = A.union(B)

    #Take the ratio of sizes
    similarity = len(nominator)/len(denominator)
    
    return similarity


similarity = jaccard_similarity(A, B)

print(similarity)


# ### Calculate Jaccard distance in Python
# 
# reference 
# [here](https://skeptric.com/jaccard-containment/)

A = {1, 2, 3, 5, 7}
B = {1, 2, 4, 8, 9}


def jaccard_distance(A, B):
    #Find symmetric difference of two sets
    nominator = A.symmetric_difference(B)

    #Find union of two sets
    denominator = A.union(B)

    #Take the ratio of sizes
    distance = len(nominator)/len(denominator)
    
    return distance


distance = jaccard_distance(A, B)
distance = jaccard_distance(A, B)

print(distance)


# ### Calculate similarity and distance of asymmetric binary attributes in Python

import numpy as np
from scipy.spatial.distance import jaccard
from sklearn.metrics import jaccard_score

A = np.array([1,0,0,1,1,1])
B = np.array([0,0,1,1,1,0])

similarity = jaccard_score(A, B)
distance = jaccard(A, B)

print(f'Jaccard similarity is equal to: {similarity}')
print(f'Jaccard distance is equal to: {distance}')


# reference 
# [here](https://python-bloggers.com/2021/12/jaccard-similarity-and-jaccard-distance-in-python/)


# ## Clustering Excercise

import random

random.seed(1410)
y = np.random.normal(0, 0.1, 50)
y = y.reshape(10, 5)
y.shape
y

scaler = StandardScaler().fit(y.T)
scaler = scaler.transform(y.T)
scaler = scaler.T
scaler.shape
scaler
scaler[0]

import statistics as stat

stat.stdev(scaler[0])

# numpy standard scaler
np.std(scaler[0])

data=[]
for n in range(scaler.shape[0]):
    data.append(np.std(scaler[n]))
#     data.append(stat.stdev(scaler[n]))

data

row = ['g1', 'g2', 'g3', 'g4', 'g5', 
       'g6', 'g7', 'g8', 'g9', 'g10']

data_y = pd.DataFrame()
data_y[0] = row
data_y[1] = data
data_y.T


# ### Euclidean distance matrix

y = pd.DataFrame(y)
y


from sklearn.metrics.pairwise import euclidean_distances

y.iloc[1:5,:3]

euclidean_distances(y.iloc[1:4,:3], y.iloc[1:4,:3])

# ### Correlation-based distance matrix

c = y.corr('pearson').T
c
c = c-1

d = c.iloc[1:5,1:5]
d

# ### Hierarchical Clustering

from scipy.cluster.hierarchy import dendrogram, linkage

linkage_data = linkage(d, method='complete', metric='euclidean')
dendrogram(linkage_data)

plt.show()

yidx = y.set_axis(row, axis=0)
yidx

linkage_data = linkage(yidx, method='complete', metric='euclidean')
dendrogram(linkage_data)

plt.show()


# ### Heatmaps

sns.clustermap(yidx)
plt.show()

# ### K-Means Fuzzy Clustering

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    y, # the data
    4, # desired number of cluster 
    2, # Array exponentiation applied to the membership function u_old at each iteration, where U_new = u_old ** m.
    error=0.005, # stopping criterion, stop early if the norm of (u[p] - u[p-1]) < error.
    maxiter=1000, # maximum number of itteration
    # maxiter=1000,
    init=None) # Initial fuzzy c-partitioned matrix.
   
cnt_df = pd.DataFrame(cntr.T)
cnt_df = round(cnt_df, 2)
cnt_df.iloc[:4,:]

cluster_membership = np.argmax(cntr, axis=0) 
cluster_membership
# the cluster is from 0~3
# so there 4 class in the 
# cluster

fanny = pd.DataFrame()
fanny[0] = row
fanny[1] = cluster_membership
fanny.T


# ### Multidimensional Scaling (MDS)


plt.figure(figsize=(8,5))
plt.scatter(loc.V1, loc.V2)

for i in range(loc.shape[0]):
    plt.text(x=loc.V1[i],# +0.3
             y=loc.V2[i],# +0.3
             s=loc.city[i], 
             fontdict=dict(color='red',size=10),
             bbox=dict(facecolor='yellow',alpha=0.5))
plt.title('cmdscale(eurodist)')
plt.show()


# ### Principal Component Analysis (PCA)

scaler = StandardScaler().fit(y)
scaler = scaler.transform(y)

dtsc = scaler
pca = PCA()
pct = pca.fit_transform(dtsc)
pct.shape


from mpl_toolkits import mplot3d

# Creating figure
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
 
# Creating plot
ax.scatter3D(pct[:, 0], pct[:, 1], pct[:, 2], color = "blue")

plt.title("PCs")
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

# show plot
plt.show()
pcr = pd.DataFrame()
pcr['pc1'] = pct[:, 0]
pcr['pc2'] = pct[:, 1]
pcr['pc3'] = pct[:, 2]
pcr

# pip install plotly

# using plotly the user interaction 
# are more active than the other python library
import plotly.express as px

df = px.data.iris()
fig = px.scatter_3d(pcr, x='pc1', y='pc2', z='pc3')
fig.show()



# 3d plotting reference 
# [here](https://www.geeksforgeeks.org/3d-scatter-plotting-in-python-using-matplotlib/) 
# [here](https://plotly.com/python/3d-scatter-plots/)
# 
# overall reference 
# [here](https://girke.bioinformatics.ucr.edu/GEN242/tutorials/rclustering/rclustering/#euclidean-distance-matrix)
