'''
Collated by Ching-Shih (Vince) Tsou 鄒慶士 博士 (Ph.D.) Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所暨智能控制與決策研究室教授), NTUB (國立臺北商業大學); Founder of the Chinese Academy of R Software (CARS) (中華R軟體學會創會理事長); Founder of the Data Science and Business Applications Association of Taiwan (DSBA) (臺灣資料科學與商業應用協會創會理事長); the Chinese Association for Quality Assessment and Evaluation (CAQAE) (中華品質評鑑協會常務監事); the Chinese Society of Quality (CSQ) (中華民國品質學會大數據品質應用委員會主任委員)
Notes: This code is provided without warranty.
'''

#### 1. Partial Least Squares Descriminant Analysis (PLS-DA)
# Partial least squares discriminant analysis (PLS-DA) is an adaptation of PLS regression methods to the problem of supervised clustering. It has seen extensive use in the analysis of multivariate datasets, such as that derived from NMR-based metabolomics. 將偏最小平方法應用於*監督式集群*，廣泛用於多變量資料集，例如：基於核磁共振代謝組學資料集

# In this method the groups within the samples are already known (e.g. experimental groups) and the goal therefore is to determine two things 已知樣本內的各群編號 —
#
# Are the groups actually different? 各群是否的確不同
# What are the features that best describe the differences between the groups? 哪些屬性最能刻畫組間的差異

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')

### Loading the data
# For this demo we will start with 1D 1H NMR data as it makes explanation and visualization of the PLS models easy to understand. 一維一小時核磁共振資料 However, later we will also generate PLS-DA models for other data types, to demonstrate how you can easily apply these same methods to any type of multivariate data set.

df = pd.read_csv('./Data/data.csv', index_col=0, header=[0,1])

df.columns
# MultiIndex(levels=[['1', '10', '11', '12', '13', '14', '15', '16', '17', '2', '3', '4', '5', '6', '7', '8', '9'], ['H', 'N']],
#           codes=[[0, 9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8], [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
#           names=['Sample', 'Group'])

df.head()


### Visualising the dataset
df.plot(kind='line', legend=False, figsize=(12,4))

df.columns.get_level_values('Group')

colormap = {
    'N': '#ff0000',  # Red
    'H': '#0000ff',  # Blue
}

colorlist = [colormap[c] for c in df.columns.get_level_values('Group')]

df.plot(kind='line', legend=False, figsize=(12,4), color=colorlist)


### Building the model
y = [g == 'N' for g in df.columns.get_level_values('Group')]
y

y = np.array(y, dtype=int)
y


from sklearn.cross_decomposition import PLSRegression

plsr = PLSRegression(n_components=2, scale=False) #1
plsr.fit(df.values.T, y) #2


### Score and weights
plsr.x_scores_
plsr.x_weights_.shape

scores = pd.DataFrame(plsr.x_scores_)
scores.index = df.columns

ax = scores.plot(x=0, y=1, kind='scatter', s=50, alpha=0.7,
                 c=colorlist, figsize=(6, 6))

ax.set_xlabel('Score on LV 1')
ax.set_ylabel('Score on LV 2')

scores

scores = pd.DataFrame(plsr.x_scores_)
scores.index = df.columns

ax = scores.plot(x=0, y=1, kind='scatter', s=50, alpha=0.7,
                 c=colorlist, figsize=(6 ,6))
ax.set_xlabel('Scores on LV 1')
ax.set_ylabel('Scores on LV 2')

for n, (x, y) in enumerate(scores.values):
    label = scores.index.values[n][0]
    ax.text(x, y, label)


f_df = df.iloc[:, df.columns.get_level_values('Sample') != '101']
f_df

f_colorlist = [colormap[c] for c in f_df.columns.get_level_values('Group')]
f_y = np.array([g == 'N' for g in f_df.columns.get_level_values('Group')], dtype=int)


from sklearn.cross_decomposition import PLSRegression

f_plsr = PLSRegression(n_components=2, scale=False)
f_plsr.fit(f_df.values.T, f_y)

f_scores = pd.DataFrame(f_plsr.x_scores_)
f_scores.index = f_df.columns

ax = f_scores.plot(x=0, y=1, kind='scatter', s=50, alpha=0.7,
                   c=f_colorlist, figsize=(6, 6))
ax.set_xlabel('Score on LV 1')
ax.set_ylabel('Score on LV 2')

for n, (x, y) in enumerate(f_scores.values):
    label = f_scores.index.values[n][0]
    ax.text(x, y, label)

samples_to_filter = ['85','101']
filter_ = [s not in samples_to_filter for s in df.columns.get_level_values('Sample')]
ff_df = df.iloc[:, filter_ ]

filter_
ff_df

# source [here](https://www.mfitzp.com/tutorials/partial-least-squares-discriminant-analysis-plsda/)


#### 2. PLS Discriminant Analysis for binary classification in Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score

# The data set for this tutorial is a bunch of NIR spectra from samples of milk powder. Milk powder and coconut milk powder were mixed in different proportions and NIR spectra were acquired. The data set is freely available for download in our Github repo.

# The data set contains 11 different classes, corresponding to samples going from 100% milk powder to 0% milk powder (that is 100% coconut milk powder) in decrements of 10%. For the sake of running a binary classification, we are going to discard all mixes except the 5th and the 6th, corresponding to the 60/40 and 50/50 ratio of milk and coconut milk powder respectively.


# Load data into a Pandas dataframe
data = pd.read_csv('./Data/milk-powder.csv')
data.columns
data.labels.value_counts() # 1 ~ 11: 20 each

# Extact first and last label in a new dataframe
binary_data = data[(data['labels'] == 5) | (data['labels'] == 6)] # (40, 603)


# Read data into a numpy array and apply simple smoothing
X_binary = savgol_filter(binary_data.values[:,2:], 15, polyorder = 3, deriv=0)
# Read categorical variables
y_binary = binary_data["labels"].values


# Map variables to 0 and 1
y_binary = (y_binary == 6).astype('uint8')


# DEfine the PLS regression object
pls_binary = PLSRegression(n_components=2)
# Fit and transform the data
X_pls = pls_binary.fit_transform(X_binary, y_binary)[0]


labplot = ["60/40 ratio", "50/50 ratio"]
# Scatter plot
unique = list(set(y_binary))
colors = [plt.cm.jet(float(i)/max(unique)) for i in unique]
 
with plt.style.context(('ggplot')):
    plt.figure(figsize=(12,10))
    for i, u in enumerate(unique):
        col = np.expand_dims(np.array(colors[i]), axis=0)
        xi = [X_pls[j,0] for j in range(len(X_pls[:,0])) if y_binary[j] == u]
        yi = [X_pls[j,1] for j in range(len(X_pls[:,1])) if y_binary[j] == u]
        plt.scatter(xi, yi, c=col, s=100, edgecolors='k',label=str(u))
 
    plt.xlabel('Latent Variable 1')
    plt.ylabel('Latent Variable 2')
    plt.legend(labplot,loc='lower left')
    plt.title('PLS cross-decomposition')
    plt.show()

# Test-train split
X_train, X_test, y_train, y_test = train_test_split(X_binary, y_binary, test_size=0.2, random_state=19)

# Define the PLS object
pls_binary = PLSRegression(n_components=2)

# Fit the training set
pls_binary.fit(X_train, y_train)

# Predictions: these won't generally be integer numbers
y_pred = pls_binary.predict(X_test)[:,0]

# "Force" binary prediction by thresholding
#binary_prediction = (pls_binary.predict(X_test)[:,0] &gt; 0.5).astype('uint8')
# invalid syntax for code up there
binary_prediction = (pls_binary.predict(X_test)[:,0]).astype('uint8')
binary_prediction

print(binary_prediction, y_test)

def pls_da(X_train,y_train, X_test):
    
    # Define the PLS object for binary classification
    plsda = PLSRegression(n_components=2)
    
    # Fit the training set
    plsda.fit(X_train, y_train)
    
    # Binary prediction on the test set, done with thresholding
    #binary_prediction = (pls_binary.predict(X_test)[:,0] &gt; 0.5).astype('uint8')
    binary_prediction = (pls_binary.predict(X_test)[:,0]).astype('uint8')
    
    return binary_prediction


accuracy = []
cval = KFold(n_splits=10, shuffle=True, random_state=19)
for train, test in cval.split(X_binary):
    
    y_pred = pls_da(X_binary[train,:], y_binary[train], X_binary[test,:])
    
    accuracy.append(accuracy_score(y_binary[test], y_pred))
 
print("Average accuracy on 10 splits: ", np.array(accuracy).mean())


# Source [here](https://nirpyresearch.com/pls-discriminant-analysis-binary-classification-python/)

#### 3. Partial Least Squares Discriminant Analysis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# READING DATA
df = pd.read_csv('./Data/seeds.csv')
df.head()
df.shape

X = df.iloc[:,0:7].values
X.shape

y_categorical = df['Name'].unique()
classes = df['Type']

# Plot settings
cols = ['blue','orange','green']
mks = ['o','^','p']
colorlist = [cols[i-1] for i in classes]
markerlist = [mks[i-1] for i in classes]

# Making the dummy Y response matrix
y = np.zeros(shape=(df.shape[0], 3))
for i in range(df.shape[0]):
    y[i, classes[i] - 1] = 1

plsr = PLSRegression(n_components=2, scale=False) # <1>
plsr.fit(X, y)
scores = plsr.x_scores_

# PCA wants a normalized Matrix
X_p = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X_p)

fig = plt.figure(figsize = (11, 5))

ax0 = fig.add_subplot(121)
for i in range(len(y_categorical)):
    indx = df.loc[df['Name'] == y_categorical[i]].index
    ax0.scatter(scores[indx,0], scores[indx,1], marker = mks[i], label = y_categorical[i])

ax0.set_xlabel('Scores on LV 1')
ax0.set_ylabel('Scores on LV 2')
ax0.set_title('PLS-DA')
ax0.legend(loc = 'upper right')

ax1 = fig.add_subplot(122)
for i in range(len(y_categorical)):
    indx = df.loc[df['Name'] == y_categorical[i]].index
    ax1.scatter(principalComponents[indx, 0], principalComponents[indx, 1], marker = mks[i], label = y_categorical[i])

ax1.set_xlabel('Component 1')
ax1.set_ylabel('Component 2')
ax1.set_title('PCA')
ax1.legend(loc = 'upper right')

# train test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

plsr = PLSRegression(n_components=2, scale=False) 
plsr.fit(X_train, y_train)

ypred = plsr.predict(X_test)

ypred.shape

# Source [here](https://data-farmers.github.io/2019-06-14-Partial-Least-Squares-Discriminant-Analysis/)

#### trying to make 3d plotting
#ypred_pd = pd.DataFrame(ypred, columns = ['A','B','C'])

#ypred_pd.head()

# markers = ['o','^','p']
# names = df['Name'].unique()

# fig = plt.figure(figsize = (18, 6))


# # SUBPLOT 1
# ax0 = fig.add_subplot(131, projection='3d')

# for i in range(3):
#     ax0.scatter(ypred_pd['A'],
#                 ypred_pd['B'], 
#                 ypred_pd['C'], 
#                 marker = markers[i], label = names[i])

# ax0.set_xlabel("A")
# ax0.set_ylabel("B")
# ax0.set_zlabel("C")
# ax0.set_title("Raw data")
# ax0.legend()

# plt.scatter(ypred_pd['A'], ypred_pd['B'])
# plt.show()

#### References:
# 1. https://www.mfitzp.com/tutorials/partial-least-squares-discriminant-analysis-plsda/
# 2. https://nirpyresearch.com/pls-discriminant-analysis-binary-classification-python/
# 3. https://data-farmers.github.io/2019-06-14-Partial-Least-Squares-Discriminant-Analysis/

