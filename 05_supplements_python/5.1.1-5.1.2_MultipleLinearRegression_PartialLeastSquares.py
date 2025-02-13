'''
Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the IDS, CICD, and CISD of NTUB (臺北商業大學資訊與決策科學研究所暨智能控制與決策研究室教授兼校務永續發展中心主任); at the ME Dept. and CAIDS of MCUT (2020年借調至明志科技大學機械工程系擔任特聘教授兼人工智慧暨資料科學研究中心主任兩年); the CSQ (2019年起任中華民國品質學會大數據品質應用委員會主任委員); the DSBA (2013年創立臺灣資料科學與商業應用協會); and the CARS (2012年創立中華R軟體學會)
Notes: This code is provided without warranty.
'''
#%%
#### 5.1.1 多元線性迴歸 (Multiple Linear Regression by OLS)
import pandas as pd
import numpy as np

solTestX = pd.read_csv('solTestX.csv',encoding='utf-8')
solTestXtrans = pd.read_csv('solTestXtrans.csv',encoding='utf-8') # 整數值變量帶小數點 Some features are binare and others are continuous
solTestY = pd.read_csv('solTestY.csv',encoding='utf-8')
solTrainX = pd.read_csv('solTrainX.csv',encoding='utf-8')
solTrainXtrans = pd.read_csv('solTrainXtrans.csv',encoding='utf-8') # 整數值變量帶小數點 Some features are binare and others are continuous
solTrainY = pd.read_csv('solTrainY.csv',encoding='utf-8')
#%%
#solTrainXtrans.columns.tolist()
#solTrainXtrans.index.tolist()
#solTrainXtrans.info()
#solTrainXtrans.describe()

len(solTrainXtrans) + len(solTestXtrans)
solTrainXtrans.shape

#### Four steps to build a regression model 數據建模四部曲：Step 1. 載入類別函數, Step 2. 宣告空模, Step 3. 傳入數據集擬合/配適實模, Step 4. 以實模進行預測與應用

from sklearn.linear_model import LinearRegression # Step 1
lm = LinearRegression() # Step 2
# From the implementation point of view, this is just plain Ordinary Least Squares (scipy.linalg.lstsq) or Non Negative Least Squares (scipy.optimize.nnls) wrapped as a predictor object.

pre = dir(lm) # 空模屬性及方法

lm.fit(solTrainXtrans, solTrainY) # Step 3 (lm的內容dir()在訓練樣本傳入配適後已發生變化！！！)

post = dir(lm) # 與pre的差異在於傳入訓練資料後計算出來的結果物件

set(post) - set(pre) # 實模與空模的差異集合 {'_residues', 'coef_', 'intercept_', 'rank_', 'singular_'} 都有下底線，Python不成文的慣例

print(lm.coef_) # 228 slope parameters
print(lm.intercept_) # only one intercepts parameters

print(lm.score(solTrainXtrans, solTrainY)) # Step 4: Returns the coefficient of determination R^2 of the prediction. (訓練集的判定係數，球員兼裁判，可能較為樂觀)

lmPred1 = lm.predict(solTestXtrans) # Step 4 預測測試樣本的可溶解度


from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

# 模型績效
r_squared = r2_score(solTestY, lmPred1) # 測試集的判定係數 
rmse = sqrt(mean_squared_error(solTestY, lmPred1))
# 印出模型績效(測試集的RMSE)
print('測試集的判定係數：{}'.format(r_squared))
print('測試集的均方根誤差：{}'.format(rmse))

#### 逐步迴歸 in Python
import stepwiseSelection as ss
# stepwiseSelection.py需放在當前工作路徑下，扮演套件/模組的角色，引入後簡稱為ss

#### 後向式迴歸(挑變量)
final_vars_b, iterations_logs_b = ss.backwardSelection(solTrainXtrans, solTrainY) # 128 + intercept (stepwiseSelection中的append可能要改寫成pd.concat()!端視pandas > 2.0.0)

#### 耗時模型，直接載入結果
import pickle # Python醃鹹菜套件
with open('final_vars_b.csv', 'wb') as f:
    pickle.dump(final_vars_b, f) # dump()表結果寫出去

with open('iterations_logs_b.csv', 'wb') as f:
    pickle.dump(iterations_logs_b, f) # dump()表結果寫出去

with open('final_vars_b.csv', 'rb') as f:
    final_vars_b = pickle.load(f) # load()表預存結果載入

#### 前向式迴歸(挑變量)
final_vars_f, iterations_logs_f = ss.forwardSelection(solTrainXtrans,solTrainY) # 55 + intercept

#### 耗時模型，直接載入結果
import pickle
with open('final_vars_f.csv', 'wb') as f:
    pickle.dump(final_vars_f, f)

with open('iterations_logs_f.csv', 'wb') as f:
    pickle.dump(iterations_logs_f, f)

with open('final_vars_f.csv', 'rb') as f:
    final_vars_f = pickle.load(f)

#### 逐步迴歸[降維]後的數據矩陣
solTrainXtrans_b = solTrainXtrans.loc[:, final_vars_b[1:]] # 1: 表不包括截距項
solTrainXtrans_f = solTrainXtrans.loc[:, final_vars_f[1:]] # 1: 表不包括截距項

#### 用statsmodels建立迴歸模型，其統計報表完整！
import statsmodels.api as sm # Step 1

lmFitAllPredictors = sm.OLS(solTrainY, solTrainXtrans).fit() # Step 2 & 3

print(lmFitAllPredictors.summary()) # 看統計報表(sklearn用Ordinary Least Squares (scipy.linalg.lstsq)計算迴歸係數，但是無法提供統計檢定結果！)

#### 用前面後向式逐步迴歸挑出的128個變量擬合模型
reducedSolMdl = sm.OLS(solTrainY,solTrainXtrans_b).fit()
print(reducedSolMdl.summary())

#### 用前面前向式逐步迴歸挑出的55個變量擬合模型
fwdSolMdl = sm.OLS(solTrainY,solTrainXtrans_f).fit()
print(fwdSolMdl.summary())
fwdSolMdl_sum = fwdSolMdl.summary()

# 檢視摘要報表的屬性與方法
[name for name in dir(fwdSolMdl_sum) if '__' not in name]

import re # re: regular expression package (Python強大的字串樣板正則表示式套件)
list(filter(lambda x: re.search(r'as', x), dir(fwdSolMdl_sum)))

#### 整個摘要報表轉為csv後存出
help(fwdSolMdl_sum.as_csv)

import pickle
with open('fwdSolMdl_sum.csv', 'wb') as f:
    pickle.dump(fwdSolMdl_sum.as_csv(), f)

# 把fwdSolMdl summary的各部分報表轉成html與DataFrame
fwdSolMdl_sum.tables # a list with three elements
len(fwdSolMdl_sum.tables) # 3
fwdSolMdl_sum.tables[0] # <class 'statsmodels.iolib.table.SimpleTable'>

#### 整體顯著性報表
fwdSolMdl_sum_as_html = fwdSolMdl_sum.tables[0].as_html()
fwdSolMdl_sum_as_html # str
pd.read_html(fwdSolMdl_sum_as_html, header=0, index_col=0)[0]

#### 模型係數顯著性報表
fwdSolMdl_sum_as_html_1 = fwdSolMdl_sum.tables[1].as_html()
pd.read_html(fwdSolMdl_sum_as_html_1, header=0, index_col=0)[0]

#### 殘差及其他統計值報表
fwdSolMdl_sum_as_html_2 = fwdSolMdl_sum.tables[2].as_html()
pd.read_html(fwdSolMdl_sum_as_html_2, header=0, index_col=0)[0]

#### ANOVA模型比較Ｆ檢定(https://www.statsmodels.org/stable/generated/statsmodels.stats.anova.anova_lm.html)
from statsmodels.stats.anova import anova_lm
anovaResults = anova_lm(fwdSolMdl, reducedSolMdl) # If None, will be estimated from the largest model. Default is None. Same as anova in R.
print(anovaResults) # 顯著(Pr(>F)很小)，故選擇後向式逐步迴歸模型

anovaResults = anova_lm(reducedSolMdl, lmFitAllPredictors)
print(anovaResults) # 不顯著(Pr(>F) = 1.0)，故選擇後向式逐步迴歸模型

#### 5.1.2 偏最小[平方/二乘]法迴歸(Partial Least Squares, PLS)
#%%
# 降維與迴歸擬合同步做，與PCR = PCA + Regression分段做是不同的！

import pandas as pd
import numpy as np
# cross decomposition交叉分解之意是降維分解時，同時考慮與y的互動是否良好
from sklearn.cross_decomposition import PLSRegression # Step 1
import matplotlib.pyplot as plt

#%%
# 記得回去讀取資料集
r2 = []
for i in np.arange(1, 51): # Try 50 ! 200
    pls = PLSRegression(n_components=i) # n_components: 1 ~ 199 or 1 ~ 50, Step 2
    pls.fit(solTrainXtrans, solTrainY) # Step 3
    score = pls.score(solTestXtrans, solTestY) # Return the coefficient of determination R^2 of the prediction. # Step 4 solTrainXtrans, solTrainY
    r2.append(score)

#%%
# 與圖5.1有異曲同工之妙
plt.plot(r2)
plt.xlabel('Number of principal components in regression')
plt.ylabel('r2 score')
plt.title('Solubility') # 看圖後約略取9或10個主成份

#%%
# 決定以九個主成份進行偏最小平方法的建模(重新擬合配適)
pls = PLSRegression(n_components=9) # Step 2
pls.fit(solTrainXtrans,solTrainY) # Step 3

#%%
# 以九個主成份的模型進行測試樣本的預測
plsPred = pls.predict(solTestXtrans) # Step 4

#%%
# 繪製預測值與實際值的散佈圖(圖5.2)
plt.scatter(solTestY, plsPred)
plt.xlabel('measured')
plt.ylabel('predicted')
plt.title('Solubility, 9 comps, test')

#%%
# 計算實際值與預測值的皮爾森相關係數
np.corrcoef(np.concatenate((solTestY.values, plsPred), axis=1), rowvar=False) # 0.93157578

# %%
# 
pls = PLSRegression(n_components=228)
pls.fit(solTrainXtrans,solTrainY)

dir(pls) # Attention to x_scores_ and y_scores_

# %%
# variance in transformed X data for each latent vector:
variance_in_x = np.var(pls.x_scores_, axis = 0, ddof=1) 
variance_in_x.sum()

# %%
# solTrainXtr is a pandas DataFrame with samples in rows and predictor variables in columns
total_variance_in_x = np.var(solTrainXtrans, axis = 0, ddof=1)
total_variance_in_x.sum()

#%%
# normalize variance by total variance:
fractions_of_explained_variance_x = variance_in_x / total_variance_in_x.sum()

fractions_of_explained_variance_x.sum()

#%%
# variance in transformed X data for each latent vector:
variance_in_y = np.var(pls.y_scores_, axis = 0, ddof=1) 

# solTrainY is one-dimensional DataFrame containing the response variable
total_variance_in_y = np.var(solTrainY, axis = 0, ddof=1)

# normalize variance by total variance:
fractions_of_explained_variance_y = variance_in_y / total_variance_in_y.sum()

fractions_of_explained_variance_y.sum()
# %%
