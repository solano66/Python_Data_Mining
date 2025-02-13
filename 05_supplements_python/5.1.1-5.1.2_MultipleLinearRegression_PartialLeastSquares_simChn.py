'''
Collated by Prof. Ching-Shih Tsou 邹庆士 教授 (Ph.D.) at the IDS, CICD, and CISD of NTUB (台北商业大学资讯与决策科学研究所暨智能控制与决策研究室教授兼校务永续发展中心主任); at the ME Dept. and CAIDS of MCUT (2020年借调至明志科技大学机械工程系担任特聘教授兼人工智慧暨资料科学研究中心主任两年); the CSQ (2019年起任品质学会大数据品质应用委员会主任委员); the DSBA (2013年创立台湾资料科学与商业应用协会); and the CARS (2012年创立中华R软体学会)
Notes: This code is provided without warranty.
'''

#### 5.1.1 多元线性回归 (Multiple Linear Regression by OLS)
import pandas as pd
import numpy as np

solTestX = pd.read_csv('solTestX.csv',encoding='utf-8')
solTestXtrans = pd.read_csv('solTestXtrans.csv',encoding='utf-8') # Some features are binary. Others are discrete or continuous.
solTestY = pd.read_csv('solTestY.csv',encoding='utf-8')
solTrainX = pd.read_csv('solTrainX.csv',encoding='utf-8')
solTrainXtrans = pd.read_csv('solTrainXtrans.csv',encoding='utf-8') # Some features are binary. Others are discrete or continuous.
solTrainY = pd.read_csv('solTrainY.csv',encoding='utf-8')

#solTrainXtrans.columns.tolist()
#solTrainXtrans.index.tolist()
#solTrainXtrans.info()
#solTrainXtrans.describe()

len(solTrainXtrans) + len(solTestXtrans)
solTrainXtrans.shape

#### Four steps to build a regression model 数据建模四部曲：Step 1. 载入类别函数, Step 2. 宣告空模, Step 3. 传入数据集拟合/配适实模, Step 4. 以实模进行预测与应用

from sklearn.linear_model import LinearRegression # Step 1
lm = LinearRegression() # Step 2
# From the implementation point of view, this is just plain Ordinary Least Squares (scipy.linalg.lstsq) or Non Negative Least Squares (scipy.optimize.nnls) wrapped as a predictor object.

pre = dir(lm) # 空模属性及方法

lm.fit(solTrainXtrans, solTrainY) # Step 3 (lm的内容在训练样本传入配适后已发生变化！！！)

post = dir(lm) # 与pre的差异在于计算出来的物件

set(post) - set(pre) # 实模与空模的差异集合 {'_residues', 'coef_', 'intercept_', 'rank_', 'singular_'}

print(lm.coef_) # 228 slope parameters
print(lm.intercept_) # only one intercepts parameters

print(lm.score(solTrainXtrans, solTrainY)) # Step 4: Returns the coefficient of determination R^2 of the prediction. (训练集的判定系数)

lmPred1 = lm.predict(solTestXtrans) # Step 4 预测测试样本的可溶解度


from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

# 模型绩效
r_squared = r2_score(solTestY, lmPred1) # 测试集的判定系数 
rmse = sqrt(mean_squared_error(solTestY, lmPred1))
# 印出模型绩效(测试集的RMSE)
print('判定系数：{}'.format(r_squared))
print('均方根误差：{}'.format(rmse))

#### 逐步回归 in Python
import stepwiseSelection as ss
# stepwiseSelection.py需放在当前工作路径下，扮演套件/模组的角色，引入后简称为ss

#### 后向式回归(挑变量)
final_vars_b, iterations_logs_b = ss.backwardSelection(solTrainXtrans, solTrainY) # 128 + intercept (stepwiseSelection中的append可能要改写成pd.concat()!端视pandas > 2.0.0)

#### 耗时模型，直接载入结果
import pickle # Python腌咸菜套件
with open('final_vars_b.csv', 'wb') as f:
    pickle.dump(final_vars_b, f) # dump()表结果写出去

with open('iterations_logs_b.csv', 'wb') as f:
    pickle.dump(iterations_logs_b, f) # dump()表结果写出去

with open('final_vars_b.csv', 'rb') as f:
    final_vars_b = pickle.load(f) # load()表预存结果载入

#### 前向式回归(挑变量)
final_vars_f, iterations_logs_f = ss.forwardSelection(solTrainXtrans,solTrainY) # 55 + intercept

#### 耗时模型，直接载入结果
import pickle
with open('final_vars_f.csv', 'wb') as f:
    pickle.dump(final_vars_f, f)

with open('iterations_logs_f.csv', 'wb') as f:
    pickle.dump(iterations_logs_f, f)

with open('final_vars_f.csv', 'rb') as f:
    final_vars_f = pickle.load(f)

#### 逐步回归[降维]后的数据矩阵
solTrainXtrans_b = solTrainXtrans.loc[:, final_vars_b[1:]] # 1: 表不包括截距项
solTrainXtrans_f = solTrainXtrans.loc[:, final_vars_f[1:]] # 1: 表不包括截距项

#### 用statsmodels建立回归模型，其统计报表完整！
import statsmodels.api as sm # Step 1

lmFitAllPredictors = sm.OLS(solTrainY, solTrainXtrans).fit() # Step 2 & 3

print(lmFitAllPredictors.summary()) # 看统计报表(sklearn用Ordinary Least Squares (scipy.linalg.lstsq)计算回归系数，但是无法提供统计检定结果！)

#### 用前面后向式逐步回归挑出的128个变量拟合模型
reducedSolMdl = sm.OLS(solTrainY,solTrainXtrans_b).fit()
print(reducedSolMdl.summary())

#### 用前面前向式逐步回归挑出的55个变量拟合模型
fwdSolMdl = sm.OLS(solTrainY,solTrainXtrans_f).fit()
print(fwdSolMdl.summary())
fwdSolMdl_sum = fwdSolMdl.summary()

# 检视摘要报表的属性与方法
[name for name in dir(fwdSolMdl_sum) if '__' not in name]

import re # re: regular expression package (Python强大的字串样板正则表示式套件)
list(filter(lambda x: re.search(r'as', x), dir(fwdSolMdl_sum)))

#### 整个摘要报表转为csv后存出
help(fwdSolMdl_sum.as_csv)

import pickle
with open('fwdSolMdl_sum.csv', 'wb') as f:
    pickle.dump(fwdSolMdl_sum.as_csv(), f)

# 把fwdSolMdl summary的各部分报表转成html与DataFrame
fwdSolMdl_sum.tables # a list with three elements
len(fwdSolMdl_sum.tables) # 3
fwdSolMdl_sum.tables[0] # <class 'statsmodels.iolib.table.SimpleTable'>

#### 整体显著性报表
fwdSolMdl_sum_as_html = fwdSolMdl_sum.tables[0].as_html()
fwdSolMdl_sum_as_html # str
pd.read_html(fwdSolMdl_sum_as_html, header=0, index_col=0)[0]

#### 模型系数显著性报表
fwdSolMdl_sum_as_html_1 = fwdSolMdl_sum.tables[1].as_html()
pd.read_html(fwdSolMdl_sum_as_html_1, header=0, index_col=0)[0]

#### 残差及其他统计值报表
fwdSolMdl_sum_as_html_2 = fwdSolMdl_sum.tables[2].as_html()
pd.read_html(fwdSolMdl_sum_as_html_2, header=0, index_col=0)[0]

#### ANOVA模型比较Ｆ检定(https://www.statsmodels.org/stable/generated/statsmodels.stats.anova.anova_lm.html)
from statsmodels.stats.anova import anova_lm
anovaResults = anova_lm(fwdSolMdl, reducedSolMdl) # If None, will be estimated from the largest model. Default is None. Same as anova in R.
print(anovaResults) # 显著(Pr(>F)很小)，故选择后向式逐步回归模型

anovaResults = anova_lm(reducedSolMdl, lmFitAllPredictors)
print(anovaResults) # 不显著(Pr(>F) = 1.0)，故选择后向式逐步回归模型

#### 5.1.2 偏最小[平方/二乘]法回归(Partial Least Squares, PLS)

# 降维与回归拟合同步做，与PCR = PCA + Regression分段做是不同的！

import pandas as pd
import numpy as np
# cross decomposition交叉分解之意是降维分解时，同时考虑与y的互动是否良好
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt

# 记得回去读取资料集
r2 = []
for i in np.arange(1, 51): # Try 50 ! 200
    pls = PLSRegression(n_components=i) # n_components: 1 ~ 199 or 1 ~ 50
    pls.fit(solTrainXtrans, solTrainY)
    score = pls.score(solTestXtrans, solTestY) # Return the coefficient of determination R^2 of the prediction. solTrainXtrans, solTrainY
    r2.append(score)

# 与图5.1有异曲同工之妙
plt.plot(r2)
plt.xlabel('Number of principal components in regression')
plt.ylabel('r2 score')
plt.title('Solubility') # 看图后约略取9或10个主成份

# 决定以九个主成份进行偏最小平方法的建模(重新拟合配适)
pls = PLSRegression(n_components=9)
pls.fit(solTrainXtrans,solTrainY)

# 以九个主成份的模型进行测试样本的预测
plsPred = pls.predict(solTestXtrans)

# 绘制预测值与实际值的散布图(图5.2)
plt.scatter(plsPred,solTestY)
plt.xlabel('measured')
plt.ylabel('predicted')
plt.title('Solubility, 9 comps, test')

# 计算实际值与预测值的皮尔森相关系数
np.corrcoef(np.concatenate((solTestY.values, plsPred), axis=1), rowvar=False) # 0.93157578

