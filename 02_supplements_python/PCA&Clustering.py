'''
Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the Dept. of ME and AI&DS (機械工程系與人工智慧暨資料科學研究中心), MCUT(明志科技大學); the IDS (資訊與決策科學研究所), NTUB (國立臺北商業大學)
Notes: This code is provided without warranty.
'''

#### Part I: Principal Component Analysis #####
#### Load digits dataset and explore it first 載入手寫數字影像，並運用主成份分析(PCA)探索之

from sklearn.datasets import load_digits # from module import classes (methods)

help(load_digits) # Help on function load_digits in module sklearn.datasets.base: Dictionary-like object, the interesting attributes are: 'data', the data to learn, 'images', the images corresponding to each sample, 'target', the classification labels for each sample, 'target_names', the meaning of the labels, and 'DESCR', the full description of the dataset.
# data used for PCA and kMeans, images used for depp learning
digits = load_digits() # data : Bunch
# data = scale(digits.data) # data scaling
type(digits) # sklearn.datasets.base.Bunch (像原生資料結構的dict)
help(digits) # Help on Bunch in module sklearn.datasets.base object, class Bunch: container object for datasets: dictionary-like object that exposes its keys as attributes

print(digits.keys()) # ['images', 'data', 'target_names', 'DESCR', 'target']
#digits["data"]

digits.images
# tmp1 = digits.images # Double clicks on tmp1 in the Variable explorer pane and explore it
digits.images.shape # (1797,8,8) will be stretched to (1797, 64) later

digits.data # same as above
# tmp2 = digits.data # Double clicks on tmp2 in the Variable explorer pane and explore it
print(digits.data.max(), digits.data.min()) # 16.0  0.0 It's a four-bit image, because 2**4 will be 16.
digits.data.shape # (1797, 64)

digits.target # array consisting of target values 0 ~ 9
# tmp3 = digits.target # Double clicks on tmp3 in the Variable explorer pane and explore it
digits.target.shape # (1797,)

digits.target_names # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
digits.target_names.shape # (10,)

digits.DESCR # Some descriptions about MNIST

#### Make some plots 顯示幾張手寫數字圖
import matplotlib.pyplot as plt # import module as a short name
plt.gray()
plt.matshow(digits.images[100]) # 2D Grey-scale image

X_digits, y_digits = digits.data, digits.target # (1797,64), (1797,)

n_row, n_col = 2, 5 # or 4, 5 for max_n = 20

# Define a function with a while loop to make several plots at one time (一次畫十張圖)
def print_digits(images, y, max_n=10):
	# Creates a new figure and set up the figure size in inches
	fig = plt.figure(figsize=(2. * n_col, 2.26 * n_row)) # define figure width and height (2. change to 1.5 if y is not shown properly)
	i = 0
	while i < max_n and i < images.shape[0]: # digits.images.shape (1797,8,8)
		p = fig.add_subplot(n_row, n_col, i + 1, xticks=[], yticks=[])
		p.imshow(images[i], cmap=plt.cm.bone,
                 interpolation='nearest') # plt.cm.bone control the colors （imshow() is similar to matshow()）
            	# label the image with the target value
		p.text(0, -1, str(y[i])) # Add annotation to plot
		i = i + 1

print_digits(digits.images, digits.target, max_n=10)

#### PCA computation
# Four steps to build our model
# Step 1. Please import the necessary class function
from sklearn.decomposition import PCA
# Step 2. We define an empty model first
estimator = PCA(n_components=64) # "PCA" is a class, instantiate an object (estimator) of class "PCA"
pre = dir(estimator) # Attributes and methods of empty model
# Steps 3 (fit). Input the data (X_digits) and fit a full-fledged (estimated, parameterized) model
# Step 4 (transform). Use the estimated model to transform the data onto a new space with 64 dimensions (i.e. the score matrix, X_pca).
X_pca = estimator.fit_transform(X_digits) # X_pca有幾維？64維 (fit for finding the rotation matrix, transform means use rotation matrix to find the score matrix)
X_pca.shape

post = dir(estimator) # Attributes and methods of full-fledged model, 'components_'(rotation/loading matrix), 'explained_variance_ratio_'(for scree plot) 

set(post) - set(pre)
# {'_fit_svd_solver',
# 'components_', # Loading matrix is here !!!
# 'explained_variance_',
# 'explained_variance_ratio_',
# 'mean_',
# 'n_components_',
# 'n_features_',
# 'n_samples_',
# 'noise_variance_',
# 'singular_values_'}

#### Visualizing all samples onto the first and second components (Score Plot 分數圖)
def plot_pca_scatter():
    colors = ['black', 'blue', 'purple', 'greenyellow', 'darkgreen', 'red', 'lime', 'cyan', 'darkorange', 'gray']
    for i in range(len(colors)):
        px = X_pca[:, 0][y_digits == i] # Coordinates of the first PC for digit "i"
        py = X_pca[:, 1][y_digits == i] # Coordinates of the second PC for digit "i"
        plt.scatter(px, py, c=colors[i])
    plt.legend(digits.target_names)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')

plot_pca_scatter() # Do you have any idea about the internal structire of images of digits

#### Another way to visualize all samples onto the first and second components (Score Plot 分數圖)
colors = ["#476A2A", "#7851B8", "#BD3430", "#4A2D4E", "#875525",
              "#A83683", "#4E655E", "#853541", "#3A3120", "#535D8E"]
plt.figure(figsize=(10, 10))
plt.xlim(X_pca[:, 0].min(), X_pca[:, 0].max()) 
plt.ylim(X_pca[:, 1].min(), X_pca[:, 1].max())

for i in range(len(digits.data)):
    # actually plot the digits as text instead of using scatter
    plt.text(X_pca[i, 0], X_pca[i, 1], str(digits.target[i]),
             color = colors[digits.target[i]],
             fontdict={'weight': 'bold', 'size': 9})
    plt.xlabel("First principal component")
    plt.ylabel("Second principal component")

#### Manifold learning with t-SNE (t-distributed Stochastic Neighbor Embedding)
# t-Distributed Stochastic Neighbor Embedding (t-SNE) is a technique for dimensionality reduction that is particularly well suited for the visualization of high-dimensional datasets.    

from sklearn.manifold import TSNE
tsne = TSNE(random_state=42)
# use fit_transform instead of fit, as TSNE has no transform method
digits_tsne = tsne.fit_transform(digits.data)

plt.figure(figsize=(10, 10))
plt.xlim(digits_tsne[:, 0].min(), digits_tsne[:, 0].max() + 1)
plt.ylim(digits_tsne[:, 1].min(), digits_tsne[:, 1].max() + 1)
for i in range(len(digits.data)):
    # actually plot the digits as text instead of using scatter
    plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str(digits.target[i]),
             color = colors[digits.target[i]],
             fontdict={'weight': 'bold', 'size': 9})
    plt.xlabel("t-SNE feature 0")
    plt.xlabel("t-SNE feature 1")

#### Visualizing the relationship between each of the first 10 components and 64 original pixel variables (Loading Plot 負荷圖)
def print_pca_components(images, n_col, n_row): # image: (64,) * 10 components
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    for i, comp in enumerate(images): # comp is an one-dimensional array (64,)
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(comp.reshape((8, 8)), interpolation='nearest') # (64,) -> (8, 8)
        plt.text(0, -1, str(i + 1) + '-component') # add annotation above
        plt.xticks(()) # remove ticks on x-axis
        plt.yticks(()) # remove ticks on y-axis

type(estimator.components_)
estimator.components_.shape # (64 components, 64 features)
estimator.components_[:10].shape # (10 components, 64 features) need to be reshaped to (10 components, 8 rows, 8 columns)

print_pca_components(estimator.components_[:10], n_col, n_row)

dir(estimator)

#### Reshape the first and the second component-feature loading/rotation matrix 將 1D (64,) 向量變形為 2D (8,8) 矩陣
estimator.components_[0]
estimator.components_[0].reshape(8,8)
#### The first component-feature loading/rotation matrix
plt.matshow(estimator.components_[0].reshape(8,8))
estimator.components_[1].reshape((8,8))

#### The second component-feature loading/rotation matrix
plt.matshow(estimator.components_[1].reshape(8,8))

#### Scree plot (陡坡圖) to determine optimal number of components (around 8 ~ 10)
import matplotlib.pyplot as plt
plt.plot(range(25), estimator.explained_variance_ratio_[:25], '-o')
plt.xlabel('# of components')
plt.ylabel('ratio of variance explained') # Find the knee area (the point that after it the marginal benefit decrease a lot, i.e. the absolute value of slope of line segments) to determine a better number of components

estimator.explained_variance_
estimator.explained_variance_ratio_

#### Model saving
import pickle
filename = 'pca_model.sav'
pickle.dump(estimator, open(filename, 'wb')) # 'wb': write out in binary form

loaded_model = pickle.load(open(filename, 'rb')) # 'rb': read something in from binary
result = loaded_model.explained_variance_ratio_
print(result)


#### Part II: Clustering Handwritten Digits with K-Means (Please try this part by yourself.) #####
from sklearn.preprocessing import scale
data = scale(digits.data) # Functional programming paradigm

digits.data

data

# print(data.mean()) # close to zero
# print(data.std()) # close to one

# By variables (64 variables)
data.mean(axis=0)
data.std(axis=0) # Attention to zeros, these zero pixel values remain zeros after scaling.

# Build training and test set
from sklearn.model_selection import train_test_split # model_selection change to cross_validation if the version of sklearn is 0.19.1

X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(data, digits.target, digits.images,  test_size=0.25, random_state=42)

n_samples, n_features = X_train.shape
import numpy as np
n_digits = len(np.unique(y_train)) # 10 classes
labels = y_train # len(y_train), 1347 labels

print_digits(images_train, y_train, max_n=20)
print_digits(images_test, y_test, max_n=20)

print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))


#### Understanding "train_test_split"
# import numpy as np
a, b = np.arange(10).reshape((5, 2)), range(5)
a
b
list(a)
list(b) # same as b
import sklearn.cross_validation
a_train, a_test, b_train, b_test = sklearn.cross_validation.train_test_split(a, b, test_size=0.33, random_state=42) # Error here 2016/2/22
a_train
b_train
a_test
b_test
#####

#### k-means clustering
# Step 1
from sklearn import cluster # comment the line with import choices module
# dir(sklearn) # more about cluster, isotonic, manifold
dir(cluster)
# help(sklearn.cluster)
help(cluster.KMeans)

# Step 2
clf = cluster.KMeans(init='k-means++', n_clusters=10, random_state=42) # 'k-means++' : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence. See section Notes in k_init for more details.
dir(clf) # Attributes and methods before fitting

# Step 3
clf.fit(X_train)
dir(clf) # Attributes and methods after fitting (Attention to attributes and methods trailing with _)
print(type(clf.labels_)) # labels_是sklearn.cluster.KMeans的屬性，另有cluster_centers_和inertia_(attribute of sklearn.cluster.KMeans is a numpy.ndarray)
print(clf.labels_.shape) # (1347,) clf.labels為訓練樣本之類別標籤
print(clf.labels_[:10])
print_digits(images_train, clf.labels_, max_n=10) # 群編號與真實數字無關！
clf.cluster_centers_ # Your clusering model is here !
clf.cluster_centers_.shape

import pandas as pd
cen = pd.DataFrame(clf.cluster_centers_)

#### Predict clusters on testing data
print(help(clf.predict))
y_pred = clf.predict(X_test)
print(len(y_pred)) # 450 test images
y_pred == 1
print(y_pred[(y_pred == 1)].shape)

#### plotting according to the clustering
def print_cluster(images, y_pred, cluster_number):
	images = images[y_pred==cluster_number]
	y_pred = y_pred[y_pred==cluster_number]
	print_digits(images, y_pred, max_n=10) # call previous print_digits()
	
for i in range(10):
	print_cluster(images_test, y_pred, i) # call print_cluster() ten times

#### Show different performance metrics, compared with "original" clusters (using the known classes)	
from sklearn import metrics

# performance evaluation, Rand index and confusion matrix
print("Adjusted rand score: {:.2}".format(metrics.adjusted_rand_score(y_test, y_pred))) # 控制到小數點下兩位
print (help(sklearn.metrics.adjusted_rand_score))

print("Homogeneity score:{:.2} ".format(metrics.homogeneity_score(y_test, y_pred)))
print("Completeness score: {:.2} ".format(metrics.completeness_score(y_test, y_pred)))
print("Confusion matrix")
print(metrics.confusion_matrix(y_test, y_pred)) # Which number is not so good?

#### PCA on X_train first
from sklearn import decomposition
pca = decomposition.PCA(n_components=2).fit(X_train)
reduced_X_train = pca.transform(X_train) # (1347,2), 物件pca的transform方法，就是傳回資料點在轉軸後的新空間座標值

# Step size of the mesh.
h = .01
# Plot the decision boundary, point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = reduced_X_train[:, 0].min() + 1, reduced_X_train[:, 0].max() - 1
y_min, y_max = reduced_X_train[:, 1].min() + 1, reduced_X_train[:, 1].max() - 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

#### understand the meshgrid
np.arange(x_min, x_max, h).shape # (1592,)
(x_max - x_min)/h # 1591.082024599925
np.arange(y_min, y_max, h).shape # (1514,)
(y_max - y_min)/h # 1513.9223624538013


xx.shape # (1514, 1592)
yy.shape # (1514, 1592)
##########################

kmeans = cluster.KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
kmeans.fit(reduced_X_train) # use the reduced (1347,2) training data to do the kmeans

Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()]) # (1514*1592,) = (2410288,)
xx.shape # (1514, 1592)
xx.ravel().shape # (2410288,) function ravel returns a flattened array
yy.shape # (1514, 1592)
yy.ravel().shape # (2410288,) function ravel returns a flattened array
np.c_[xx.ravel(), yy.ravel()].shape # (2410288,2), np.c_[] translates slice objects to concatenation along the second axis

help(kmeans.predict) # Predict the closest cluster each sample in X belongs to

############### about the numpy.ravel function (untangle)
x = np.array([[1, 2, 3], [4, 5, 6]])
x.shape
print(np.ravel(x))
print(x.reshape(-1)) # sample as above
###############

# Put the result into a color plot
Z = Z.reshape(xx.shape) # from (2410288,) to (1514, 1592)
plt.figure(1)
plt.clf() # Clear the current figure
plt.imshow(Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()), cmap=plt.cm.Paired, aspect='auto', origin='lower') # heat map

# scatter plot of PCA-reduced data
plt.plot(reduced_X_train[:, 0], reduced_X_train[:, 1], 'k.', markersize=2)

centroids = kmeans.cluster_centers_
centroids.shape # (10, 2)

# Plot the centroids as a white X
plt.scatter(centroids[:, 0], centroids[:, 1], marker='.', s=169, linewidths=3, color='w', zorder=10) # s: size in points

plt.title('K-means clustering on the digits dataset (PCA reduced data)\nCentroids are marked with white dots')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

help(plt.scatter)

##### Please ignore the followings #####
#### Affinity Propagation
aff = cluster.AffinityPropagation()
help(cluster.AffinityPropagation) # class AffinityPropagation(sklearn.base.BaseEstimator, sklearn.base.ClusterMixin)

aff.fit(X_train) # X_train's shape is (1347, 64)
help(aff.fit) # Help on method fit in module sklearn.cluster.affinity_propagation_
help(aff) # does not have a method of predict

print(aff.cluster_centers_indices_.shape) # (112,)
help(aff) # has attribute of cluster_centers_indices_, labels_, affinity_matrix_

#### Mean Shift
#ms = cluster.MeanShift()
#ms.fit(X_train)
#print(ms.cluster_centers_.shape) # (18,64)


#### Gaussian Mixture Models, similar to k-means
from sklearn import mixture
gm = mixture.GMM(n_components=n_digits, covariance_type='tied', random_state=42) # covariance_type = 'tied': each pixel is expected to be related
gm.fit(X_train)
help(gm) # has a method of predict


# Print train clustering and confusion matrix
y_pred = gm.predict(X_test)
y_pred
y_pred.shape # (450,)

print("Adjusted rand score:{:.2}".format(metrics.adjusted_rand_score(y_test, y_pred))) # 0.65, better than k-menas (0.59)
print("Homogeneity score:{:.2}".format(metrics.homogeneity_score(y_test, y_pred))) # 0.74 (unsupervised version of precision, greater is better)
print("Completeness score: {:.2}".format(metrics.completeness_score(y_test, y_pred))) # 0.79 (unsupervised version of recall, greater is better)
########################################
