#### Principal Component Analysis (PCA)
# 
# The [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) model is part of a group of techniques based on [modeling on latent variables](https://en.wikipedia.org/wiki/Latent_variable) , that is, that effect transformations on the data in order to express them as a new set of variables that explain hidden (latent) characteristics of the original data. Depending on which characteristics are desired to be explained, several techniques can be proposed.
# 
# In particular, the PCA model aims to *generate an orthogonal set of linear combinations of the original variables, with the aim of selecting a subset of these combinations that appropriately summarizes the variability of the data*.
# 
# Let's define a simple two-dimensional dataset to see the model in action:


# example adapted from https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html

import numpy as np
rng = np.random.RandomState(1)
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T

import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1])
plt.axis('equal')
plt.xlabel('x1')
plt.ylabel('x2');


# In `scikit-learn `, PCA is available in module [sklearn.decomposition](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition).
# 
# Initializing the model and applying it to the `X` set shown above:

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)


# PCA generates the transformed data by projecting the original data in special directions called *principal components*. Each of these directions, which are orthogonal to each other, explains a portion of the data variance.
# 
# In `scikit-learn`, the main components are stored in the `components_` attribute and the explained variances in the `explained_variance_` attribute:


print('components:')
print(pca.components_)

print('\nexplained variances:')
print(pca.explained_variance_)


# Note that the first principal component explains a much larger portion of the variance than the second.
# 
# To visualize the principal components, let's define the following function, which plots vectors on a graph:


def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)


# Visualize:


plt.scatter(X[:, 0], X[:, 1], alpha=0.3)

for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v)
    
plt.axis('equal')
plt.xlabel('x1')
plt.ylabel('x2');


# In the figure above, it is possible to visualize what we had observed numerically with the `explained_variance_` attribute: the first principal component explains a much larger portion of the variance than the second.
# 
# In fact, the first principal component identifies the *direction of greatest variability of the data*. The second principal component, being orthogonal to the first, covers a much smaller portion of the variability.
# 
# So far we have used the `fit` method, which only calculates the principal components. To project the data into the principal components and get the transformed data matrix `T`, one must use the `transform` function:

# 先座標軸旋轉
T = pca.transform(X)


# Plotting the transformed data (latent variables):


plt.scatter(T[:, 0], T[:, 1], alpha=0.3)
plt.axis('equal')
plt.xlabel('t1')
plt.ylabel('t2');


# Basically, the PCA transformation rotated the dataset, aligning the direction of greatest variability of the data with the $x$ axis of Cartesian space.
# 
# A big advantage of expressing data in terms of `T` is that, unlike the columns of the original matrix `X`, the columns of `T` are *linearly independent*. That is, there is no correlation between the columns, which in theory eliminates information redundancy: each column contains its own information, independent of the others.
# 
# We have just seen that, from a geometric point of view, the PCA generates the transformed data `T` by projecting the original data `X` in the directions defined by the principal components. It is also useful to interpret this operation from an algebraic point of view: the PCA generates the transformed data `T` by linearly combining the columns of the original data of `X`. The coefficients of linear combinations are given by the principal components. In our example, the components are:


pca.components_.round(3)


# So, if $\mathbf{x}_1$ and $\mathbf{x}_2$ are the columns of the matrix $\mathbf{X} = [\mathbf{x}_1, \mathbf{x}_2]$, the columns of the transformed matrix $\mathbf{T} = [\mathbf{t}_1, \mathbf{t}_2]$ are:
# 
# $$
# \mathbf{t}_1 = -0.944\mathbf{x}_1 - 0.329\mathbf{x}_2
# $$
# $$
# \mathbf{t}_2 = -0.329\mathbf{x}_1 + 0.944\mathbf{x}_2
# $$
# 
# This is one of the beauties of Linear Algebra: concepts can be interpreted from both a geometric and algebraic point of view.

#### PCA - Dimensionality Reduction 再降維
# 
# In the example above, we apply PCA to generate two main components, specifying the hyperparameter `n_components = 2` at the model initialization stage. It is the maximum number of components that can be generated in this case, or the number of variables in the original set is 2.
# 
# But, since the first component captures almost all the variability of two dice, how about using just the second one and discarding it? This procedure, in which the transformed dice show a smaller dimension than two original dice, is called [dimensionality reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction).
# 
# For example, applying o PCA to generate just 1 main component:


pca = PCA(n_components=1)
pca.fit(X)

T = pca.transform(X)


# The original set `X` has two variables, while the transformed set `T` has only one:


# display(X.shape)
# display(T.shape)


# Plotting the `T` set:


plt.plot(T, np.zeros(len(T)),'.')
plt.gca().get_yaxis().set_visible(False)
plt.xlabel('t1');


# Now the data is represented in only one dimension! All the variability of the other dimension was lost, but as it was a small part of the total variability, the loss was not very significant.
# 
# Furthermore, in many applications, *our goal is precisely to discard this part of the information*. For example, when it represents measurement noise! In this sense, PCA can be seen as a technique to *filter noise*.
# 
# Dimensionality reduction is also useful for visualization purposes. For example, if a dataset has more than three dimensions, we cannot visualize it in the plane or in Cartesian space; however, we can plot the first two or three latent variables and visualize most of the variability in the data (this will become clear in Hands-on 2, proposed later).

#### PCA - Reconstruction
# 
# The `T` transformed data can be reprojected into the original space using the `inverse_transform` function:


X_reconstruido = pca.inverse_transform(T) # Back to (200, 2)


# The operation of reprojecting the data into the original space is known as *reconstruction*.
# 
# Plotting, for comparison, the original and the reconstructed set:

# A beautiful plot !
plt.scatter(X[:, 0], X[:, 1], alpha=0.2, label = 'original')
plt.scatter(X_reconstruido[:, 0], X_reconstruido[:, 1], alpha=0.8, label='reconstrução')
plt.legend()
plt.axis('equal')
plt.xlabel('x1')
plt.ylabel('x2');


# Note that, although the transformed set `T` has dimension 1, the reconstruction has dimension 2, as well as the original set:


display(X.shape) # (200, 2)
display(T.shape) # (200, 1)
display(X_reconstruido.shape) # (200, 2)

err = X - X_reconstruido

# That is, the `X_reconstructed` dataset is in the original dimensionality of the problem, but with one big difference: without the information initially present along the second principal component.

# ***Hands on 2!***
# 
# * Use module [sklearn.datasets](https://scikitlearn.org/stable/modules/classes.html#module-sklearn.datasets) to load [Fisher's Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set). Research the set (its origin, the problem it proposes to illustrate, etc.). What is its original dimensionality? Apply the PCA model to the data. Check the variance explained by each principal component. Plot the first two dimensions of the transformed data (that is, the projections of the original data onto the first two principal components) on a Cartesian chart.
# 
# What conclusions can this procedure suggest about the nature of the data?