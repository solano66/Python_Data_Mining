'''
Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the Dept. of ME and AI&DS (機械工程系與人工智慧暨資料科學研究中心主任), MCUT(明志科技大學); the IDS (資訊與決策科學研究所), NTUB (國立臺北商業大學); the CARS(中華R軟體學會創會理事長); and the DSBA(臺灣資料科學與商業應用協會創會理事長)
Notes: This code is provided without warranty.
'''

import numpy as np # for everything
import matplotlib.pyplot as plt # for DV
from sklearn import datasets # for importing 'iris' dataset
from sklearn.decomposition import PCA # for PCA
import pandas as pd # for 'DataFrame'
from sklearn.preprocessing import StandardScaler # for Standardization before doing the PCA

iris = datasets.load_iris()
X = iris.data
y = iris.target

# Remember to scale the data before do the PCA.
scaler = StandardScaler()
scaler.fit(X) # 計算各變量的平均數與標準差
X_std = scaler.transform(X) # 套用前面計算的平均數與標準差，轉換 X 中的原始值 
pca = PCA() # try n_components=2 and compare x_new and X_std
x_new = pca.fit_transform(X_std)

def biplot(score, coeff, labels = None): # labels is text for original variables
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min()) # for normalization
    scaley = 1.0/(ys.max() - ys.min()) # for normalization
    # Score plot first
    plt.scatter(xs * scalex, ys * scaley, c = y) # attention to normalization and c argument
    # Then loading/rotation plot
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1], color = 'r', alpha = 0.5) # from origin (0, 0) to end of the arrows
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center') # attention to 1.15
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center') # attention to 1.15
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid() # add grid lines

# Call the function. Use only the 2 PCs.
biplot(x_new[:,0:2], np.transpose(pca.components_[0:2, :]), labels=['SepalLength','SepalWidth','PetalLength','PetalWidth'])
plt.show()

dir(pca)

pca.components_

pca.explained_variance_
pca.explained_variance_ratio_








