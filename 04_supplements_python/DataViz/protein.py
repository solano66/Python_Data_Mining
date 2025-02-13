'''
Collated by Ching-Shih (Vince) Tsou 鄒慶士 博士 (Ph.D.) Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所暨智能控制與決策研究室教授), NTUB (國立臺北商業大學); Founder of the Chinese Academy of R Software (CARS) (中華R軟體學會創會理事長); Founder of the Data Science and Business Applications Association of Taiwan (DSBA) (臺灣資料科學與商業應用協會創會理事長); the Chinese Association for Quality Assessment and Evaluation (CAQAE) (中華品質評鑑協會常務監事); the Chinese Society of Quality (CSQ) (中華民國品質學會大數據品質應用委員會主任委員)
Notes: This code is provided without warranty.
'''

import pandas as pd

dir(pd)
help(pd.read_csv)

food = pd.read_csv('./data/protein.csv')
food.iloc[:, 1:].apply(sum, axis=1) # Not sum up to 1, but it's still a close data

dir(food)

food.dtypes
tmp = food.describe(include='all').T

food.info()

# Model building four steps
# This is an object-oriented way to standardize the data.
# Step 1 (Attention! This is a class not a function in Python.)
from sklearn.preprocessing import StandardScaler

# Step 2
sc = StandardScaler()

# Step 3 (loc and iloc, what's the difference?)
sc.fit(food.iloc[:,1:])

# Step 4
food_z = sc.transform(food.iloc[:,1:]) 

food_z = pd.DataFrame(food_z, columns = food.columns[1:])

tmp1 = pd.DataFrame(food_z).describe().T

# Step 1
from sklearn.cluster import KMeans

# Step 2
# km = KMeans(n_clusters = 7, init = 'random', n_init=10)
km = KMeans(n_clusters = 7, init = 'random', n_init=25)

# Steps 3 & 4
km.fit_transform(food_z) # Compute clustering and transform X to cluster-distance space. X transformed in the new space.

dir(km)

km.cluster_centers_
km.labels_

#### 結果如何呈現很重要！表 + 圖 How to show your clustering results. Please make a table and a plot.
tbl = pd.DataFrame({'Country': food['Country'], 'Cluster': km.labels_})
tbl.sort_values(by = 'Cluster')

import matplotlib.pyplot as plt
plt.scatter(food.loc[:,' RedMeat'], food.loc[:,' WhiteMeat'], c=km.labels_)
# plt.text() can only do it one by one
for i in range(food.shape[0]):
    plt.text(food.loc[:,' RedMeat'][i], food.loc[:,' WhiteMeat'][i], food['Country'].values[i]) # Possible six clusters is better.
   
# Sum of distances of samples to their closest cluster center (Total sum of squares within cluster) (**R is more stable than Python**) ####
dir(km)
km.inertia_

### Practice 1:
# Please use k-menas clustering to separate 25 countries into 5, 6, 7 groups, and try to explain your results. (Don't forget to standardize your data before clustering them.)

### Practice 2:
# Before you do the clustering, please find the first two principal components first, and then do the k-means clsutering on the reduced two-dimensional space. Compare the clustering results with the previous one.

# SPSS: Statistical Packages for Social Science (its original name), Statistical Product and Service Solutions
# It's not so good for DS & AI.
# To learn the Python and R. They are proper tools for Big Data.









