'''
Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the IDS, CICD, CISD, and CCE of NTUB (臺北商業大學資訊與決策科學研究所暨智能控制與決策研究室教授兼校務永續發展中心主任&推廣教育部主任); at the ME Dept. and CAIDS of MCUT (2020年借調至明志科技大學機械工程系擔任特聘教授兼人工智慧暨資料科學研究中心主任兩年); the CSQ (2019年起任中華民國品質學會AI暨大數據品質應用委員會主任委員); the DSBA (2013年創立臺灣資料科學與商業應用協會); and the CARS (2012年創立中華R軟體學會)
Notes: This code is provided without warranty.
'''
#%%
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
# from mlxtend.data import iris_data
from sklearn.model_selection import train_test_split
#%%
# x, y = iris_data()
x, y = load_iris(return_X_y=True)
lr = LogisticRegression(max_iter=150)
dt = DecisionTreeClassifier(max_depth = 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
lr.fit(x_train, y_train)
dt.fit(x_train, y_train)

s1 = lr.score(x_test, y_test)
s2 = dt.score(x_test, y_test)

print('Model A accuracy: %.2f%%' % (s1*100))
print('Model B accuracy: %.2f%%' % (s2*100))
#%%
#### K-fold cross-validated [paired t test] for classifiers or regressors comparison
from mlxtend.evaluate import paired_ttest_kfold_cv # conda install mlxtend --channel conda-forge (https://rasbt.github.io/mlxtend/installation/)

t, p = paired_ttest_kfold_cv(estimator1=lr,
                              estimator2=dt,
                              X=x, y=y,
                              random_seed=1)

alpha = 0.05

print('t statistic: %.3f' % t)
print('aplha ', alpha)
print('p value: %.3f' % p)

if p > alpha:
  print("Fail to reject null hypotesis")
else:
  print("Reject null hypotesis")
#%%
#### Resampled [paired t test] for classifiers or regressors comparison
from mlxtend.evaluate import paired_ttest_resampled

t, p = paired_ttest_resampled(estimator1=lr,
                              estimator2=dt,
                              X=x, y=y,
                              random_seed=1)

alpha = 0.05

print('t statistic: %.3f' % t)
print('aplha ', alpha)
print('p value: %.3f' % p)

if p > alpha:
  print("Fail to reject null hypotesis")
else:
  print("Reject null hypotesis")
#%%
#### 5x2cv [paired t test] for classifiers or regressors comparison
from mlxtend.evaluate import paired_ttest_5x2cv

t, p = paired_ttest_5x2cv(estimator1=lr,estimator2=dt,X=x, y=y)

alpha = 0.05

print('t statistic: %.3f' % t)
print('aplha ', alpha)
print('p value: %.3f' % p)

if p > alpha:
  print("Fail to reject null hypotesis")
else:
  print("Reject null hypotesis")

# All results are significant.

#### Reference:
# https://rasbt.github.io/mlxtend/user_guide/evaluate/paired_ttest_kfold_cv/
# https://rasbt.github.io/mlxtend/user_guide/evaluate/paired_ttest_resampled/
# https://rasbt.github.io/mlxtend/user_guide/evaluate/paired_ttest_5x2cv/

# %%
