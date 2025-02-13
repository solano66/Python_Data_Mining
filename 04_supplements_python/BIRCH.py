#### BIRCH Clustering Algorithm Example In Python
#%%
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
from sklearn.datasets import make_blobs
from sklearn.cluster import Birch
#%%
X, clusters = make_blobs(n_samples=450, centers=6, cluster_std=0.70, random_state=0)
plt.scatter(X[:,0], X[:,1], alpha=0.7, edgecolors='b')
#%%
#### Initialize and train our model, using the following parameters:
# threshold: The radius of the subcluster obtained by merging a new sample and the closest subcluster should be lesser than the threshold.
# branching_factor: Maximum number of CF subclusters in each node
# n_clusters: Number of clusters after the final clustering step, which treats the subclusters from the leaves as new samples. If set to None, the final clustering step is not performed and the subclusters are returned as they are.

brc = Birch(branching_factor=50, n_clusters=None, threshold=1.5)
brc.fit(X)
#%%
labels = brc.predict(X)
#%%
plt.scatter(X[:,0], X[:,1], c=labels, cmap='rainbow', alpha=0.7, edgecolors='b')

#### Final Thoughts
# BIRCH provides a clustering method for very large datasets. It makes a large clustering problem plausible by concentrating on densely occupied regions, and creating a compact summary. BIRCH can work with any given amount of memory, and the I/O complexity is a little more than one scan of data. Other clustering algorithms can be applied to the subclusters produced by BIRCH.

#### Reference:
# BIRCH Clustering Algorithm Example In Python (https://medium.com/p/fb9838cbeed9)
# %%
