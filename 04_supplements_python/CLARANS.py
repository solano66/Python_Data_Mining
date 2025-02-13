#%%
#### Partitional Clustering using CLARANS (Clustering Large Applications based on RANdomized Search) Method with Python Example

# 1. Overview of Clustering.
# Clustering is a form of unsupervised learning because in such kind of algorithms class label is not present.

# In general, clustering is the process of partitioning a set of data objects into subsets. Where each subset is a cluster, such that objects in a cluster are similar to one another, yet dissimilar to objects in other clusters.

# Based on the characteristics of the algorithms, there are 4 main types of clustering techniques:


# CLARANS is a type of Partitioning method.

# 2. Brief Description of Partitioning Methods.
# Partitioning methods are the most fundamental type of cluster analysis, they organize the objects of a set into several exclusive group of clusters (i.e each object can be present in only one group).

# Partitioning algorithms require the number of clusters ( k ) as it’s starting point.

# Thus given a dataset D, consisting of n points, and k (k << n), partitioning algorithm organizes the objects into k partitions (clusters).

# The clusters are formed by optimizing an objective partitioning criterion, such as a dissimilarity function based on distance, so that the objects within a cluster are “similar” to one another and “dissimilar” to objects in other clusters in terms of the data set attributes.

# CLARANS is a type of Partitioning method.

# 3. Comparison of Partitioning Methods.

# K-means: The k-means algorithm defines the centroid of a cluster as the mean value of the points within the cluster.

# That is why K-means is sensitive to noise and outliers because a small number of such data can substantially influence the mean value.

# 3.1 — K-medoids: To overcome the problem of sensitivity to outliers, instead of taking the mean value as the centroid, we can take actual data point to represent the cluster, this is what K-medoids does.

# But the k-medoids methods is very expensive when the dataset and k value is large.

# 3.2 — CLARA: To scale up the K-medoids method, CLARA was introduced. CLARA does not take the whole dataset into consideration instead uses a random sample of the dataset, from which the best medoids are taken.

# But the effectiveness of CLARA depends on the sample size. CLARA cannot find a good clustering if any of the best sampled medoids is far from the best k-medoids.

# 3.3 — CLARANS (Clustering Large Applications based upon RANdomized Search) : It presents a trade-off between the cost and the effectiveness of using samples to obtain clustering.

# 4. Overview of CLARANS:
# It presents a trade-off between the cost and the effectiveness of using samples to obtain clustering.

# First, it randomly selects k objects in the data set as the current medoids. It then randomly selects a current medoid x and an object y that is not one of the current medoids.

# Then it checks for the following condition:

# Can replacing x by y improve the absolute-error criterion?

# If yes, the replacement is made. CLARANS conducts such a randomized search l times. The set of the current medoids after the l steps is considered a local optimum.

# CLARANS repeats this randomized process m times and returns the best local optimal as the final result.

from pyclustering.cluster.clarans import clarans; # pip install pyclustering
from pyclustering.utils import timedcall;
from sklearn import datasets
#%%
#import iris dataset from sklearn library
iris =  datasets.load_iris();

#get the iris data. It has 4 features, 3 classes and 150 data points.
data = iris.data
#%%
"""!
The pyclustering library clarans implementation requires
list of lists as its input dataset. 套件需要輸入巢狀串列
Thus we convert the data from numpy array to list.
"""
data = data.tolist()

#get a glimpse of dataset
print("A peek into the dataset : ",data[:4])
#%%
"""!
@brief Constructor of clustering algorithm CLARANS.
@details The higher the value of maxneighbor, the closer is CLARANS to K-Medoids, and the longer is each search of a local minima.
@param[in] data: Input data that is presented as list of points (objects), each point should be represented by list or tuple.
@param[in] number_clusters: amount of clusters that should be allocated.
@param[in] numlocal: the number of local minima obtained (amount of iterations for solving the problem).
@param[in] maxneighbor: the maximum number of neighbors examined.        
"""
clarans_instance = clarans(data, number_clusters=3, numlocal=6, maxneighbor=4);
#%%
#calls the clarans method 'process' to implement the algortihm
(ticks, result) = timedcall(clarans_instance.process);
print("Execution time : ", ticks, "\n");
#%%
#returns the clusters 
clusters = clarans_instance.get_clusters();
#%%
#returns the mediods 
medoids = clarans_instance.get_medoids();
#%%
print("Index of the points that are in a cluster : ",clusters)
print("The target class of each datapoint : ",iris.target)
print("The index of medoids that algorithm found to be best : ",medoids)

#### Reference:
# https://medium.com/analytics-vidhya/partitional-clustering-using-clarans-method-with-python-example-545dd84e58b4
# %%
