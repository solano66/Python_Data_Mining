'''
Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the IDS, CICD, and CISD of NTUB (臺北商業大學資訊與決策科學研究所暨智能控制與決策研究室教授兼校務永續發展中心主任); at the ME Dept. and CAIDS of MCUT (2020年借調至明志科技大學機械工程系擔任特聘教授兼人工智慧暨資料科學研究中心主任兩年); the CSQ (2019年起任中華民國品質學會大數據品質應用委員會主任委員); the DSBA (2013年創立臺灣資料科學與商業應用協會); and the CARS (2012年創立中華R軟體學會)
Notes: This code is provided without warranty.
'''

#### 5.1.3 脊迴歸、     LASSO迴歸與彈性網罩懲罰模型(Ridge Regression, LASSO, and Elastic Nets)
from sklearn import model_selection
from sklearn.linear_model import Ridge, ElasticNet, Lasso # Step 1

# 隨機打亂樣本順序(shuffle=True, random_state=1)後，切分為十等分(n_splits=10)
kf_10 = model_selection.KFold(n_splits=10, shuffle=True, random_state=1)

#### Ridge Regression (調lambda或alpha)
mse = []
for i in np.linspace(start=0, stop=0.008, num=5): # i here means lambda (alpha in Python)
    rr = Ridge(alpha=i) # Step 2
    score = -1 * model_selection.cross_val_score(rr, solTrainXtrans, solTrainY, cv=kf_10, scoring='neg_mean_squared_error').mean() # All scorer objects follow the convention that [higher return values are better] than lower return values. Thus metrics which measure the distance between the model and the data, like metrics.mean_squared_error, are available as neg_mean_squared_error which return the negated value of the metric. (scikit-learn特異之處) Step 3 & 4
    mse.append(score)

plt.plot(np.sqrt(mse)) # lambda* = 0.00714286
plt.xlabel(str(np.linspace(start=0, stop=0.008, num=5)))
# Refit the Ridge model under lambda = 0.00714286, and write the regression equation you get.

min(mse)
np.linspace(start=0, stop=0.008, num=5)[mse.index(min(mse))]

rr = Ridge(alpha=0.00714286)
rr.fit(solTrainXtrans, solTrainY)
dir(rr)
rr.n_features_in_
rr.coef_
rr.intercept_
# rr.sparse_coef_ # AttributeError: 'Ridge' object has no attribute 'sparse_coef_'

rr.coef_.shape # (1, 228)
sum(rr.coef_.reshape((-1,)) != 0) # 228

ls = Lasso(alpha=0.00714286) # alpha=0.1
ls.fit(solTrainXtrans, solTrainY)
ls.n_features_in_
ls.coef_
ls.intercept_
ls.sparse_coef_ # 70 versus 6

ls.coef_.shape # (228,)
sum(ls.coef_ != 0)

#### Elastic Nets (兩個參數要調)
# Step 1
from sklearn.model_selection import GridSearchCV
# Step 2
lm_elastic = ElasticNet(max_iter = 5000)

parameters = {'alpha':[0.01, 0.05, 0.1], 'l1_ratio':np.linspace(0.05, 1, num=20)}

# Step 2
search = GridSearchCV(lm_elastic, parameters, scoring='neg_mean_squared_error', cv=10)

# Step 3
search.fit(solTrainXtrans, solTrainY)

print(search.best_score_, search.best_params_) # {'alpha': 0.01, 'l1_ratio': 0.05}
# Refit the Elastic Nets model under l'alpha': 0.01, 'l1_ratio': 0.05 and write the regression equation you get.

#### 5.1.4 線性判別分析(Linear Discriminant Analysis, LDA)
import numpy as np
import pandas as pd
mu1 = [1,1]
mu2 = [3.5,2]
sig = [[1,0.85],[0.85,2]] # Variance-Covariance Matrix
n1 = 1000*0.9
n2 = 1000*0.1
group = np.append(np.repeat(0,n1),np.repeat(1,n2))

np.random.seed(130)
X1train = pd.DataFrame(np.random.multivariate_normal(mu1, sig, 900))
X2train = pd.DataFrame(np.random.multivariate_normal(mu2, sig, 100))
Xtrain = pd.concat([X1train, X2train], ignore_index=True)
Xtrain['group'] = group
dtrain = Xtrain
dtrain.iloc[897:903,:]

np.random.seed(131)
X1test = pd.DataFrame(np.random.multivariate_normal(mu1, sig, 900))
X2test = pd.DataFrame(np.random.multivariate_normal(mu2, sig, 100))
Xtest = pd.concat([X1test, X2test], ignore_index=True)
Xtest['group'] = group
dtest = Xtest
dtest.iloc[897:903,:]

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure()
plt.subplot(1,2,1)
sns.scatterplot(x=dtrain[0],y=dtrain[1],hue=group) # x=, y=, 2023/June/16
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Training data')
plt.subplot(1,2,2)
sns.scatterplot(x=dtest[0],y=dtest[1],hue=group)
plt.xlabel('X1')
#plt.ylabel('X2')
plt.title('Test data')
plt.show()
# ValueError: 'color' kwarg must be an mpl color spec or sequence of color specs.
# For a sequence of values to be color-mapped, use the 'c' argument instead.


dtrain_X = dtrain.iloc[:,0:2]
dtrain_y = dtrain.iloc[:,2]
dtest_X = dtest.iloc[:,0:2]
dtest_y = dtest.iloc[:,2]

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis().fit(dtrain_X, dtrain_y)
lda.score(dtrain_X,dtrain_y)
ldapred = lda.predict(dtest_X)
pd.crosstab(dtest_y,ldapred)
np.unique(ldapred, return_counts=True)


from sklearn.metrics import accuracy_score
print(accuracy_score(dtest_y,ldapred)*100)

#### 5.1.5 羅吉斯迴歸分類(Logistic Regression)與廣義線性模型(Generalized Linear Models)
import statsmodels.api as sm
glm_binom = sm.GLM(dtrain_y, dtrain_X, family=sm.families.Binomial(sm.genmod.families.links.logit()))
res = glm_binom.fit()
print(res.summary())
predLogit = res.predict(dtest_X)
predLabel = predLogit > 0.5

pd.crosstab(dtest_y,predLabel)
print(accuracy_score(dtest_y,predLabel)*100)

