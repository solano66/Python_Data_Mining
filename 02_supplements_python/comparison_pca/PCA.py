import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

cell = pd.read_csv('cell_num.csv', index_col=0)

from sklearn.preprocessing import StandardScaler

x = StandardScaler().fit_transform(cell)
x = pd.DataFrame(x, columns=cell.columns)

from sklearn.decomposition import PCA

pcm = PCA()
pca1 = pcm.fit_transform(x) # score matrix

dir(pcm)

rotation = pcm.components_

print('pca variance with 5 PCs: ',pcm10.explained_variance_)
print('pca variance ratio with 5 PCs: ',pcm10.explained_variance_ratio_)

figure(figsize=(8, 6), dpi=80)
plt.scatter(pca1[:, 0], pca1[:, 1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

def myplot(score,coeff,labels=None):
    figure(figsize=(8, 6), dpi=80)
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley,s=5)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'green', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
 
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()

myplot(pca1[:,0:2],np.transpose(pcm10.components_[0:2, :]),list(x.columns))
plt.show()


