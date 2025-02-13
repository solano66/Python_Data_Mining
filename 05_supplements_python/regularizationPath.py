"""
Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the IDS and CICD (資訊與決策科學研究所暨智能控制與決策研究室), Director of the Center for Institutional and Sustainable Development (校務永續發展中心主任), NTUB (國立臺北商業大學)
Notes: This code is provided without warranty.
"""

import mglearn as ml # !pip install mglearn
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from numpy import genfromtxt

# (506, 105) target included
dataset = genfromtxt('https://raw.githubusercontent.com/m-mehdi/tutorials/main/boston_housing.csv', delimiter=',')
X = dataset[:,:-1]
y = dataset[:,-1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)

lr = LinearRegression().fit(X_train, y_train)

print(f"Linear Regression-Training set score: {lr.score(X_train, y_train):.2f}")
print(f"Linear Regression-Test set score: {lr.score(X_test, y_test):.2f}")

from sklearn.linear_model import Ridge
ridge = Ridge(alpha=0.7).fit(X_train, y_train)
print(f"Ridge Regression-Training set score: {ridge.score(X_train, y_train):.2f}")
print(f"Ridge Regression-Test set score: {ridge.score(X_test, y_test):.2f}")

from sklearn.linear_model import Lasso
lasso = Lasso(alpha=1.0).fit(X_train, y_train)
print(f"Lasso Regression-Training set score: {lasso.score(X_train, y_train):.2f}")
print(f"Lasso Regression-Test set score: {lasso.score(X_test, y_test):.2f}")

print(f"Number of features: {sum(lasso.coef_ != 0)}")

# This means that only 4 of the 104 features in the training set are used in the lasso regression model, while the rest are ignored.

# Let's adjust alpha to reduce underfitting by decreasing its value to 0.01:

lasso = Lasso(alpha=0.01).fit(X_train, y_train)
print("Lasso Regression-Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
print("Lasso Regression-Test set score: {:.2f}".format(lasso.score(X_test, y_test)))

print(f"Number of features: {sum(lasso.coef_ != 0)}")

from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=0.01, l1_ratio=0.01).fit(X_train, y_train) # l2_ratio=0.99
print(f"Elastic Net-Training set score: {elastic_net.score(X_train, y_train):.2f}")
print(f"Elastic Net-Test set score: {elastic_net.score(X_test, y_test):.2f}")

####################
# This needs to be instantiated outside the loop so we don't start from scratch each time.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X_train = pd.DataFrame(X_train)

# lr = LinearRegression(C = 1,  # we'll override this in the loop
#                         warm_start=True, # warm_startbool, default=False. When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution. Useless for liblinear solver.
#                         fit_intercept=True,
#                         solver = 'liblinear',
#                         penalty = 'l2',
#                         tol = 0.0001,
#                         n_jobs = -1,
#                         verbose = -1,
#                         random_state = 0
#                        )
lr = Lasso(alpha=0.01)
counter = 0
for c in np.arange(-10, 2, dtype=np.float): # np.float64
    lr.set_params(alpha=10**c) # C -> alpha
    model=lr.fit(X_train, y_train)


    coeff_list=model.coef_.ravel()
    
    if counter == 0:
        coeff_table = pd.DataFrame(pd.Series(coeff_list,index=X_train.columns),columns=[10**c])
    else:
        temp_table = pd.DataFrame(pd.Series(coeff_list,index=X_train.columns),columns=[10**c])
        coeff_table = coeff_table.join(temp_table,how='left')
    counter += 1

plt.rcParams["figure.figsize"] = (20,10)
coeff_table.transpose().iloc[:,:10].plot()
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.legend(loc='right')
plt.xscale('log')
plt.show()


# In general, to avoid overfitting, the regularized models are preferable to a plain linear regression model. In most scenarios, ridge works well. But in case you're not certain about using lasso or elastic net, elastic net is a better choice because, as we've seen, lasso removes strongly correlated features.

#### Conclusion
# This tutorial explored different ways of avoiding overfitting in linear machine learning models. We discussed why overfitting happens and what ridge, lasso, and elastic net regression methods are. We also applied these techniques to the Boston housing dataset and compared the results. Some other techniques, such as **early stop and dropout**, can be used for regularizing complex models, while the latter is mainly used for regularizing artificial neural networks.

#### Reference:
# https://www.dataquest.io/blog/regularization-in-machine-learning/
# https://datascience.stackexchange.com/questions/103405/elegant-way-to-plot-the-l2-regularization-path-of-logistic-regression-in-python
