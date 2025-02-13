'''
Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the IDS and CICD of NTUB (國立臺北商業大學資訊與決策科學研究所暨智能控制與決策研究室教授); at the ME Dept. and CAIDS of MCUT (2020~2022借調至明志科技大學機械工程系任特聘教授兼人工智慧暨資料科學研究中心主任); the CSQ (2019年起任中華民國品質學會大數據品質應用委員會主任委員); the DSBA (2013年創立臺灣資料科學與商業應用協會); and the CARS (2012年創立中華R軟體學會)
Dataset: algae.csv
Notes: This code is provided without warranty.
'''

#### 3.1 隨機誤差模型 How to find a robust random error or probabilitic model ?
import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

algae = pd.read_csv('algae.csv', encoding='utf-8')

# 資料分析工作中，經常需要將文字資料轉為數值(又稱編碼encoding or coding)，以利後續的數學建模，常用的編碼方式有標籤編碼(label encoding)、單熱編碼(one-hot encoding)、以及統計學中的虛擬編碼(dummy encoding)。R語言讀入資料集後可以自動進行標籤編碼，亦即將字串自動轉換為因子，而Python常須手動編碼字串變量，其中套件scikit-learn偏好單熱編碼，R語言則偏好標籤及虛擬編碼。 We have to map the categorical variables to integers.

algae['season'] = algae['season'].map({'spring':1,'summer':2,'autumn':3,'winter':4}).astype(int) # label encoding(Python dict其實就是一個對應關係)
algae['size'] = algae['size'].map({'small':1,'medium':2,'large':3}).astype(int)
algae['speed'] = algae['speed'].map({'low':1,'medium':2,'high':3}).astype(int)

# 各變量遺缺狀況統計表
algae.isnull().sum() # sum(is.na(algae)) in R, 是把二維表中所有真假值加總起來(rowSums()與columnSums())

# 各樣本遺缺狀況統計表
algae.isnull().sum(axis=1)

# 遺缺值超過4個變量的樣本編號
algae.isnull().sum(axis='columns')[algae.isnull().sum(axis='columns') > 4]

# 移除遺缺程度嚴重的樣本
cleanAlgae_tmp = algae.dropna(axis='rows',thresh=13) # 變數個數大於等於13者留之

# 以各變項中位數填補遺缺值 Imputations on missing data
cleanAlgae = pd.DataFrame()
for col in cleanAlgae_tmp.columns:
    cleanAlgae[col] = cleanAlgae_tmp[col].fillna(cleanAlgae_tmp[col].median()) # skipna=True

type(cleanAlgae_tmp.columns)

# algae['mxPH'].median()

# 確認資料表中已無遺缺值
cleanAlgae.isnull().sum()

# 繪製散佈圖，探索變量關係(不置入書中)
# targets = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7']
# predictors = ['season','size','speed','mxPH','mnO2','Cl','NO3','NH4','oPO4','PO4','Chla']

# import matplotlib.pyplot as plt
# import seaborn as sns

# for feature in cleanAlgae.columns[:11]:
#     p = sns.scatterplot(data=cleanAlgae.loc[:, predictors+['a1']], x=feature, y='a1', palette='RdBu')
#     p.set(xscale='log', yscale='log')
#     plt.show()

# 選擇X變數以及y變數 Select the features and target
X = cleanAlgae[['season','size','speed','mxPH','mnO2','Cl','NO3','NH4','oPO4','PO4','Chla']]
y = cleanAlgae[['a1']]

# 切割訓練集與測試集(亂數種子1234) Use the simple train-test split (under random seed 1234)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=50, random_state=1234)

# 以148個訓練樣本估計函數關係
#### 線性迴歸法一：使用scikit-learn.linear_model的LinearRegression類別 (by scipy.linalg.lstsq 未提供統計檢定報表)
# 註：如用梯度陡降法逼近迴歸係數，也是沒有檢定報表！
# Step 1 載入建模所需類別函數 Import the necessary libraries
from sklearn.linear_model import LinearRegression

# Step 2 宣告空模(假設線性，但參數未知的模型)
a1Lm = LinearRegression() # 一切照預設設定
pre = dir(a1Lm) # 空模屬性與方法

# Step 3 將訓練樣本傳入，配適/擬合模型參數，空模轉為實模
a1Lm.fit(X_train, y_train)
post = dir(a1Lm)
set(post) - set(pre) # Python不成文的規定，資料傳入配適模型完成後，新增的屬性與方法多帶下底線！請留意intercept_和coef_

# 11個變數的迴歸係數(為何不是15個？類別變數未虛擬編碼！)
a1Lm.coef_ # 11個斜率係數，Why? without dummy coding (one-hot encoding) for categorical variables
a1Lm.coef_.shape

# 迴歸模型配適完後的屬性與方法
# 截距係數
a1Lm.intercept_

# 特徵矩陣X的秩
a1Lm.rank_

# 特徵矩陣X的奇異值(https://baike.baidu.hk/item/%E5%A5%87%E7%95%B0%E5%80%BC/9975162)
a1Lm.singular_

# 特徵矩陣X的特徵個數
a1Lm.n_features_in_

# 特徵矩陣X的特徵名稱
a1Lm.feature_names_in_

# 取得模型的參數
a1Lm.get_params()

# Step 4 擬合完成後運用模型a1Lm 估計訓練樣本的a1 有害藻類濃度
trainPred = a1Lm.predict(X_train)

# 訓練樣本的模型績效指標RMSE值(參見3.2.1節)
from sklearn import metrics
trainRMSE = np.sqrt(metrics.mean_squared_error(y_train, trainPred))
print("訓練樣本的RMSE值為 The RMSE for 148 training samples：{}".format(trainRMSE))

# Step 4 以模型a1Lm估計測試樣本的a1有害藻類濃度
testPred = a1Lm.predict(X_test)
# 測試樣本的模型績效指標RMSE值
testRMSE = np.sqrt(metrics.mean_squared_error(y_test, testPred))
print("測試樣本的RMSE值為 A more reliable estimate of RMSE is from the test set：{}".format(testRMSE))

algae[['a1']].describe()

# 下面是測試集的RMSE比訓練集RMSE還低的結果(亂數種子20531) I want to show you if I change the random seed from 1234 to 20531, then the metrics RMSE are upside down ! 
np.random.seed(20531)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# 以重新切分後的148個訓練樣本估計函數關係
from sklearn.linear_model import LinearRegression
a1Lm2 = LinearRegression() # a1Lm2 = LinearRegression().fit(X_train, y_train)
a1Lm2.fit(X_train, y_train)

dir(a1Lm2)
# 11個變數的迴歸係數(類別變數未虛擬編碼！)
a1Lm2.coef_ # without dummy coding (one-hot encoding) for categorical variables
# 截距係數
a1Lm2.intercept_

# 擬合完成後運用模型a1Lm2估計訓練樣本的a1有害藻類濃度
trainPred = a1Lm2.predict(X_train)


# 訓練樣本的模型績效指標RMSE值(參見3.2.1節)
from sklearn import metrics
trainRMSE = np.sqrt(metrics.mean_squared_error(y_train,trainPred))
print("訓練樣本的RMSE值為 The RMSE for 148 training samples under 20531：{}".format(trainRMSE))

# 以模型a1Lm2估計測試樣本的a1有害藻類濃度
testPred = a1Lm2.predict(X_test)
# 測試樣本的模型績效指標RMSE值
testRMSE = np.sqrt(metrics.mean_squared_error(y_test,testPred))
print("測試樣本的RMSE值為 A more reliable estimate of RMSE is from the test set：{}，低於訓練樣本的RMSE值{}".format(testRMSE, trainRMSE))
# 怎麼會這樣！？亂數種子好恐怖也！其實隨機誤差模型是依訓練樣本配適均值模型，要找到穩健性或魯棒性(robust/robustness)高的模型，其解決之道是反覆進行多次實驗(eg. **repeated** train-test split/hold-out重複多次的保留法, bootstrapping拔靴抽樣法, or 10-fold cross-validation時折交叉驗證法)
# Different random seeds get different results ! Which one should we rely on? The answer is that we need to conduct repeated experiments again and again. And average out the performance across the experiments to understand the variation/confidence behind the probabilistic models, in order to get a robust one.

#### 遺憾之處：沒看到因子變量的虛擬編碼、迴歸係數的t檢定、模型整體配適度F檢定、殘差檢定！

#### 線性迴歸法二：使用統計報表較完整的statsmodels套件(Why? Because of R.)(用矩陣代數求解迴歸係數)
# 語法一：R model formula (數學統計人群)
# 為了後續使用 model formula (統計慣用的建模語法，來自R語言)
cleanAlgae_train = pd.concat([X_train, y_train], axis='columns')
# Step 1
import statsmodels.formula.api as smf
# Steps 2 & 3
# ols stands for Ordinary Least Squares
a1Lm3 = smf.ols('a1 ~ season + size + speed + mxPH + mnO2 + Cl + NO3 + NH4 + oPO4 + PO4 + Chla', data=cleanAlgae_train).fit()
# SyntaxError: invalid syntax in the shorthand . in R

dir(a1Lm3)
a1Lm3.summary() # 有一點失望～(沒看到因子變量的虛擬編碼)

type(a1Lm3.summary()) # statsmodels.iolib.summary.Summary

results_summary = a1Lm3.summary()
# Note that tables is a list. The table at index 1 is the "core" table. Additionally, read_html puts dfs in a list, so we want index 0

type(results_summary.tables)
len(results_summary.tables)

results_as_html = results_summary.tables[0].as_html()
pd.read_html(results_as_html, header=0, index_col=0)[0]

results_as_html = results_summary.tables[1].as_html()
pd.read_html(results_as_html, header=0, index_col=0)[0]

results_as_html = results_summary.tables[2].as_html()
pd.read_html(results_as_html, header=0, index_col=0)[0]

# 語法二：statsmodels (丟屬性矩陣X_train及反應變數向量y_train，計算機人群)
import statsmodels.api as sm
a1Lm4 = sm.OLS(y_train, X_train).fit()

dir(a1Lm4)
a1Lm4.summary() # 報表同上


#### 線性迴歸法三：如何產生與R語言或統計書上相同的虛擬變數編碼報表 Dummy coding in Python {statsmodels}
# Search terms by 'statsmodel dummy coding'
# https://douglaspsteen.github.io/handling_categorical_variables_with_statsmodels_ols
import pandas as pd
# *資料要重新讀入*
algae = pd.read_csv('algae.csv', encoding='utf-8', dtype={'season': 'category', 'size': 'category', 'speed': 'category'}) # pandas的'category'類似R的'factor'

algae.dtypes # 留意前三欄為category型別(原來為object)

# 移除遺缺的樣本
cleanAlgae2 = algae.dropna(axis='rows') # (184, 18) 此處不將重度遺缺(直接刪除)和輕度遺缺(用中位數填補)樣本分開處理

cleanAlgae2.dtypes # The first three are not objects again ! They are category.

# 選擇X變數以及y變數
X = cleanAlgae2[['season','size','speed','mxPH','mnO2','Cl','NO3','NH4','oPO4','PO4','Chla']]
y = cleanAlgae2[['a1']]

from sklearn.model_selection import train_test_split

# 切割訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=50, random_state=1234)

import statsmodels.formula.api as smf

# Please use C() to enclose your category variables. *模型公式符號中請用C()包住類別變數(虛擬編碼)*
f_rev = 'a1 ~ C(season) + C(size) + C(speed) + mxPH + mnO2 + Cl + NO3 + NH4 + oPO4 + PO4 + Chla'

cleanAlgae2_train = pd.concat([X_train, y_train], axis=1)

model_rev = smf.ols(formula=f_rev, data=cleanAlgae2_train).fit()
model_rev.summary()


# 擬合完成後運用模型a1Lm2估計訓練樣本的a1有害藻類濃度
trainPred = model_rev.predict(X_train)

# 訓練樣本的模型績效指標RMSE值(參見3.2.1節)
from sklearn import metrics
trainRMSE = np.sqrt(metrics.mean_squared_error(y_train,trainPred))
print("訓練樣本的RMSE值為：{}".format(trainRMSE))

# 以模型a1Lm2估計測試樣本的a1有害藻類濃度
testPred = model_rev.predict(X_test)
# 測試樣本的模型績效指標RMSE值
testRMSE = np.sqrt(metrics.mean_squared_error(y_test,testPred))
print("測試樣本的RMSE值為：{}，高於訓練樣本的RMSE值{}".format(testRMSE, trainRMSE))


#### Difference between statsmodel OLS (為了解釋，細節較多) and scikit-learn linear regression (為了準確預測，一定準確嗎？)
# https://stats.stackexchange.com/questions/146804/difference-between-statsmodel-ols-and-scikit-linear-regression
# Statsmodels follows largely the traditional model where we want to know how well a given model fits the data, and what variables "explain" or affect the outcome, or what the size of the effect is. Scikit-learn follows the machine learning tradition where the main supported task is chosing the "best" model for prediction. 配適狀況是否良好、解釋或影響結果(因)變量的(自)變數有哪些、或影響的大小; 預測最佳的模型

#### Interpreting Linear Regression Through statsmodels.summary()
# https://medium.com/swlh/interpreting-linear-regression-through-statsmodels-summary-4796d359035a
# https://www.adrian.idv.hk/2021-07-16-statsmodels/

# Df Residuals is another name for our Degrees of Freedom in our mode. This is calculated in the form of ‘n-k-1’ or ‘number of observations-number of predicting variables-1.’ Df Model numbers our predicting variables. If you’re wondering why we only entered 3 predicting variables into the formula but both Df Residuals and Model are saying there are 6, we’ll get into this later. Our Covariance Type is listed as nonrobust. Covariance is a measure of how two variables are linked in a positive or negative manner, and a robust covariance is one that is calculated in a way to minimize or eliminate variables, which is not the case here.

# Log-likelihood is a numerical signifier of the likelihood that your produced model produced the given data. 模型產生給定資料的可能性 It is used to compare coefficient values for each variable in the process of creating the model. AIC and BIC are both used to compare the efficacy of models in the process of linear regression, using a penalty system for measuring multiple variables. These numbers are used for feature selection of variables.

# Omnibus describes the normalcy of the distribution of our residuals using skew and kurtosis as measurements. 利用偏態與峰度檢驗殘差的常態性 A 0 would indicate perfect normalcy. Prob(Omnibus) is a statistical test measuring the probability the residuals are normally distributed. A 1 would indicate perfectly normal distribution. Skew is a measurement of symmetry in our data, with 0 being perfect symmetry. Kurtosis measures the peakiness of our data, or its concentration around 0 in a normal curve. Higher kurtosis implies fewer outliers.

# Durbin-Watson is a measurement of homoscedasticity, or an even distribution of errors throughout our data. (i.e. autocorrelation in the residuals) Heteroscedasticity would imply an uneven distribution, for example as the data point grows higher the relative error grows higher. Ideal homoscedasticity will lie between 1 and 2. 

# Jarque-Bera (JB) and Prob(JB) are alternate methods of measuring the same value as Omnibus and Prob(Omnibus) using skewness and kurtosis. We use these values to confirm each other. 

# Condition number 條件數 is a measurement of the sensitivity of our model as compared to the size of changes in the data it is analyzing. 模型敏感度，亦即數據變動後模型的變動幅度(https://blog.csdn.net/lanchunhui/article/details/51372831) Multicollinearity is strongly implied by a high condition number. 高條件數顯示自變量間存在多重共線性 Multicollinearity a term to describe two or more independent variables that are strongly related to each other and are falsely affecting our predicted variable by redundancy.

