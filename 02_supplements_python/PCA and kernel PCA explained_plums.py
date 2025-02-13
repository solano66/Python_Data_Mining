'''
Collated by Ching-Shih (Vince) Tsou 鄒慶士 博士 (Ph.D.) Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所暨智能控制與決策研究室教授), NTUB (國立臺北商業大學); Founder of the Chinese Academy of R Software (CARS) (中華R軟體學會創會理事長); Founder of the Data Science and Business Applications Association of Taiwan (DSBA) (臺灣資料科學與商業應用協會創會理事長); the Chinese Association for Quality Assessment and Evaluation (CAQAE) (中華品質評鑑協會常務監事); the Chinese Society of Quality (CSQ) (中華民國品質學會大數據品質應用委員會主任委員)
Notes: This code is provided without warranty.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
from sklearn.preprocessing import StandardScaler, KernelCenterer # KernelCenterer centers the features without explicitly computing the mapping. Working with centered kernels is sometime expected when dealing with algebra computation such as eigendecomposition for KernelPCA for instance. 核函數中心化函數在未實質轉換下，進行變量的中心化(參見SVM該節)
from sklearn.decomposition import PCA, KernelPCA
from sklearn.utils import extmath
from sklearn.metrics.pairwise import euclidean_distances


#### PCA by user-defined function

def pca(X, n_components=2):
    
    # Preprocessing - Standard Scaler
    X_std = StandardScaler().fit_transform(X)
    
    # Calculate covariance matrix
    cov_mat = np.cov(X_std.T) # rowvar=True
    
    # Get eigenvalues and eigenvectors
    eig_vals, eig_vecs = np.linalg.eigh(cov_mat) 
    
    # flip eigenvectors' sign to enforce deterministic output
    eig_vecs, _ = extmath.svd_flip(eig_vecs, np.empty_like(eig_vecs).T)
    
    # Concatenate the eigenvectors corresponding to the highest n_components eigenvalues
    matrix_w = np.column_stack([eig_vecs[:,-i] for i in range(1,n_components+1)])
    
    # Get the PCA reduced data
    Xpca = X_std.dot(matrix_w)
 
    return Xpca

# We are going to decompose NIR spectral data from fresh plums. 近紅外線光譜
data = pd.read_csv('./data/plums.csv').iloc[:,1:]
X = data.values
Xstd = StandardScaler().fit_transform(X)

##### Scikit-learn PCA
pca1 = PCA(n_components=2)
Xpca1 = pca1.fit_transform(X)

#### Our implementation
Xpca2 = pca(X, n_components=2)

np.array_equal(Xpca1, Xpca2) # False

#### Comparison
with plt.style.context(('ggplot')):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
 
    #plt.figure(figsize=(8,6))
    ax[0].scatter(Xpca1[:,0], Xpca1[:,1], s=100, edgecolors='k')   
    ax[0].set_xlabel('PC 1')
    ax[0].set_ylabel('PC 2')
    ax[0].set_title('Scikit learn')
    
    ax[1].scatter(Xpca2[:,0], Xpca2[:,1], s=100, facecolor = 'b', edgecolors='k')
    ax[1].set_xlabel('PC 1')
    ax[1].set_ylabel('PC 2')
    ax[1].set_title('Our implementation')
    plt.show()


#### KPCA by user-defined function

def ker_pca(X, n_components=3, gamma = 0.01):
    
    # Calculate euclidean distances of each pair of points in the data set
    dist = euclidean_distances(X, X, squared=True)
    
    # Calculate Gaussian kernel matrix
    K = np.exp(-gamma * dist)
    Kc = KernelCenterer().fit_transform(K)
    
    # Get eigenvalues and eigenvectors of the kernel matrix
    eig_vals, eig_vecs = np.linalg.eigh(Kc)
    
    # flip eigenvectors' sign to enforce deterministic output
    eig_vecs, _ = extmath.svd_flip(eig_vecs, np.empty_like(eig_vecs).T)
    
    # Concatenate the eigenvectors corresponding to the highest n_components eigenvalues
    Xkpca = np.column_stack([eig_vecs[:,-i] for i in range(1,n_components+1)])
 
    return Xkpca

##### Scikit-learn KernelPCA
kpca1 = KernelPCA(n_components=3, kernel='rbf', gamma=0.01)
Xkpca1 = kpca1.fit_transform(Xstd)

#### Our implementation
Xkpca2 = ker_pca(Xstd)

np.array_equal(Xkpca1, Xkpca2) # False

#### Comparison
with plt.style.context(('ggplot')):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
 
    #plt.figure(figsize=(8,6))
    ax[0].scatter(Xkpca1[:,0], Xkpca1[:,1], s=100, edgecolors='k')   
    ax[0].set_xlabel('PC 1')
    ax[0].set_ylabel('PC 2')
    ax[0].set_title('Scikit learn')
    
    ax[1].scatter(Xkpca2[:,0], Xkpca2[:,1], s=100, facecolor = 'b', edgecolors='k')
    ax[1].set_xlabel('PC 1')
    ax[1].set_ylabel('PC 2')
    ax[1].set_title('Our implementation')
    plt.show()


#### refrence [here](https://nirpyresearch.com/pca-kernel-pca-explained/)
