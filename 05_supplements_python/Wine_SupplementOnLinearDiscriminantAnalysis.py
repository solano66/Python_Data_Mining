'''
Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the IDS and CICD of NTUB (國立臺北商業大學資訊與決策科學研究所暨智能控制與決策研究室教授); at the ME Dept. and CAIDS of MCUT (2020~2022借調至明志科技大學機械工程系任特聘教授兼人工智慧暨資料科學研究中心主任); the CSQ (2019年起任中華民國品質學會大數據品質應用委員會主任委員); the DSBA (2013年創立臺灣資料科學與商業應用協會); and the CARS (2012年創立中華R軟體學會) 
Notes: This code is provided without warranty.
'''

#### 5.1.4 線性判別分析補充範例
#### Practice 1:
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
wine = pd.read_csv('Wine.csv')
wine.shape
wine.dtypes
#Alcohol                 酒精
#Malic_Acid              蘋果酸,羥基丁二酸
#Ash                     灰
#Ash_Alcanity            灰堿度
#Magnesium               鎂
#Total_Phenols           酚類
#Flavanoids              黃酮類化合物
#Nonflavanoid_Phenols    非黃酮類酚類
#Proanthocyanins         花青素
#Color_Intensity         顏色強度
#Hue                     色彩色度
#OD280                   
#Proline                 [生化]脯氨酸
#Customer_Segment        顧客區隔

X = wine.iloc[:, 0:13].values
y = wine.iloc[:, 13].values
np.unique(y, return_counts=True) # Three classes

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_train.shape # (142, 13)
X_test.shape # (36, 2)
y_train.shape # (142,)
y_test.shape # (36,)

# Feature Scaling
wine.describe(include='all').T
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test) # fitted by X_train and apply to both sets 

# Applying LDA for Dimensionality Reduction
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

# Fitting Logistic Regression to the Reduced Training Set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()
plt.show()

#### Practice 2:
# Importing the dataset again
A = pd.read_csv('Wine.csv')
# Get the targets (first column of file)
y = A.iloc[:, -1]
# Remove targets from input data
A = A.iloc[:, :-1]

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(A, y)
drA = lda.transform(A)

dir(lda)
lda.scalings_.shape # (13, 2)
lda.coef_.shape # lda.coef_
lda.means_.shape # (3, 13)
lda.xbar_.shape # (13,)
lda.priors_.shape # (3,)

# Slope and intercept of the discriminant functions
sv = lda.coef_  @ lda.scalings_ # (3, 2)
c  = np.dot(lda.means_ - lda.xbar_, lda.scalings_) # (3, 2)
iv = -.5 * np.square(c).sum(1) + np.log(lda.priors_) # (3, )
# Slope and intercepts for decision boundaries
m  = -sv[:, 0] / sv[:, 1] # (3, )
b  =  iv       / sv[:, 1] # (3, )


# Points where discriminant functions equal == decision boundaries
es = np.vstack((sv[[0]] - sv[[1]],
                sv[[1]] - sv[[2]],
                sv[[2]] - sv[[0]]))
ei = np.array([iv[0] - iv[1],
               iv[1] - iv[2],
               iv[2] - iv[0]])
# Slope and intercepts for decision boundaries
m    = -es[:, 0] / es[:, 1] 
b    =  ei       / es[:, 1]

# Data extracted; perform LDA
lda = LinearDiscriminantAnalysis()

# from sklearn.model_selection import KFold
# kf = KFold(n_splits=3, random_state=0, shuffle=True)

from sklearn.model_selection import StratifiedKFold
stratified_folder = StratifiedKFold(n_splits=3, random_state=0, shuffle=True)

print('LDA Results: ')
#for trn, tst in kf.split(A):
for trn, tst in stratified_folder.split(A, y):
    lda.fit(A.values[trn], y.values[trn])
    #Compute classification error
    outVal = lda.score(A.values[tst], y.values[tst])
    print('Score: {:.2%}'.format(outVal))


#### Reference:
# https://nicholastsmith.wordpress.com/2016/02/13/wine-classification-using-linear-discriminant-analysis-with-python-and-scikit-learn/





