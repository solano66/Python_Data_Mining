#### Principal Components Regression in Python

#### Step 1: Import Necessary Packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 
from sklearn import model_selection
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#### Step 2: Load the Data
#define URL where data is located
url = "https://raw.githubusercontent.com/Statology/Python-Guides/main/mtcars.csv"

#read in data
data_full = pd.read_csv(url)

#select subset of data
data = data_full[["mpg", "disp", "drat", "wt", "qsec", "hp"]]

#view first six rows of data
data[0:6]

#### Step 3: Fit the PCR Model

#define predictor and response variables
X = data[["mpg", "disp", "drat", "wt", "qsec"]]
y = data[["hp"]]

#scale predictor variables
pca = PCA()
X_reduced = pca.fit_transform(scale(X))

#define cross validation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

regr = LinearRegression() # ols in statsmodels
mse = []

# Calculate MSE with only the intercept
score = -1*model_selection.cross_val_score(regr,
           np.ones((len(X_reduced),1)), y, cv=cv,
           scoring='neg_mean_squared_error').mean()    
mse.append(score)

# Calculate MSE using cross-validation, adding one component at a time
for i in np.arange(1, 6):
    score = -1*model_selection.cross_val_score(regr,
               X_reduced[:,:i], y, cv=cv, scoring='neg_mean_squared_error').mean()
    mse.append(score)
    
# Plot cross-validation results    
plt.plot(mse)
plt.xlabel('Number of Principal Components')
plt.ylabel('MSE')
plt.title('hp')

# Thus, the optimal model includes just the first two principal components.

np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

#### Step 4: Use the Final Model to Make Predictions

#split the dataset into training (70%) and testing (30%) sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0) 

#scale the training and testing data
X_reduced_train = pca.fit_transform(scale(X_train))
X_reduced_test = pca.transform(scale(X_test))[:,:1]

#train PCR model on training data 
regr = LinearRegression()
regr.fit(X_reduced_train[:,:1], y_train)

#calculate RMSE
pred = regr.predict(X_reduced_test)
np.sqrt(mean_squared_error(y_test, pred))

# We can see that the test RMSE turns out to be 40.2096. This is the average deviation between the predicted value for hp and the observed value for hp for the observations in the testing set.

#### Here is a dataset probably more suitable to apply the PCR = PCA + Regression approach

solTestX = pd.read_csv('solTestX.csv',encoding='utf-8')
solTestXtrans = pd.read_csv('solTestXtrans.csv',encoding='utf-8') # 整數值變量帶小數點 Some features are binare and others are continuous
solTestY = pd.read_csv('solTestY.csv',encoding='utf-8')
solTrainX = pd.read_csv('solTrainX.csv',encoding='utf-8')
solTrainXtrans = pd.read_csv('solTrainXtrans.csv',encoding='utf-8') # 整數值變量帶小數點 Some features are binare and others are continuous
solTrainY = pd.read_csv('solTrainY.csv',encoding='utf-8')

#solTrainXtrans.columns.tolist()
#solTrainXtrans.index.tolist()
#solTrainXtrans.info()
#solTrainXtrans.describe()

len(solTrainXtrans) + len(solTestXtrans)
solTrainXtrans.shape

#### Reference:
# https://www.statology.org/principal-components-regression-in-python/