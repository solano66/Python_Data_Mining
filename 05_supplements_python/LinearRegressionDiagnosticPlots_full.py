'''
Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the Dept. of ME and AI&DS (機械工程系與人工智慧暨資料科學研究中心), MCUT(明志科技大學); the IDS (資訊與決策科學研究所), NTUB (國立臺北商業大學)
Notes: This code is provided without warranty.
'''

#### Linear Regression 線性迴歸
#### Case One 案例一
#### Packages importing
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
# Two APIs in statsmodels packages (statsmodels的兩種api)
import statsmodels.api as sm
import statsmodels.formula.api as smf # f stands for formula (~, +, ....)

# There are lots of advanced regression models in statsmodels (LOESS, LOcally Estimated Scatterplot Smoothing) and LOWESS (LOcally WEighted Scatterplot Smoothing)) 進階迴歸方法
from statsmodels.nonparametric.smoothers_lowess import lowess
import numpy as np

#### The focuses between statsmodels and scipy.stats are different. 兩套件焦點不同！
# There is another statistics module in scipy
# The former focus on regression modeling, and the latter is more about probability distribution and statistical tests. 迴歸建模 versus 機率分佈與統計檢定

import scipy.stats as stats
# from pydataset import data # Access and load many dataset # !conda install pydataset --y
# pip install pydataset

#### Data Understanding and Preprocessing 資料理解與整理
# mtcars = data('mtcars')
mtcars = pd.read_csv('mtcars.csv', index_col=[0])
mtcars.head()

# PyDataset Documentation (adopted from R Documentation. The displayed examples are in R)
# data('mtcars', show_doc = True)
# [, 1]	mpg	Miles/(US) gallon
# [, 2]	cyl	Number of cylinders
# [, 3]	disp	Displacement (cu.in.) 排氣量
# [, 4]	hp	Gross horsepower
# [, 5]	drat	Rear axle ratio 後軸比
# [, 6]	wt	Weight (1000 lbs)
# [, 7]	qsec	1/4 mile time (time of the car speed from 0 tp 1/4 mile)
# [, 8]	vs	Engine (0 = V-shaped, 1 = straight)
# [, 9]	am	Transmission (0 = automatic, 1 = manual) 變速箱
# [,10]	gear	Number of forward gears 前進檔
# [,11]	carb	Number of carburetors 化油器

#### Linear regression model fitting and summary report 線性迴歸模型配適及結果報表
# The formula API function smf.ols() requires two inputs, the formula for producing the best fit line, and the dataset. 運用statsmodels的formula API建模及估計迴歸係數
model = smf.ols(formula='mpg ~ cyl + wt', data=mtcars) # Specify your model 建立模型
results = model.fit() # Use fit() method to estim ate model coefficients 估計係數

dir(results) # can see the method 'summary' 有看到summary方法
print(results.summary())

# There are three parts in the ouput summary report from OLS 報表包含三部份
#### The first part on model overall significance 第一部分是關於模型整體的顯著性
# The top of our summary starts by giving us a few details we already know. Our Dependent Variable is ‘mpg,’ we’ve using OLS known as Ordinary Least Squares 普通最小平方/二乘法, and the Date and Time we’ve created the Model. Next, it details our Number of Observations 樣本數 in the dataset. Df Residuals is another name for our Degrees of Freedom 殘差自由度 in our mode (32 - 2 - 1). This is calculated in the form of ‘n-k-1’ or ‘number of observations-number of predicting variables-1.’ Df Model numbers our predicting variables. 模型自由度表入模自變數個數

# Our Covariance Type is listed as nonrobust. 共變異數計算方式為非穩健的 Covariance is a measure of how two variables are linked in a positive or negative manner, 共變異數說明兩變量以何種方式互動 (正向或是負向) and a robust covariance is one that is calculated in a way to minimize or eliminate variables, which is not the case here. 穩健的共變異數估計方法需要刪除變量

# 依據判定係數(小於1)、F檢定統計量(越大越好)、F檢定的p值(< 0.05)、概似率/比之對數值、AIC及BIC訊息準則判斷模型整體是否顯著！？
# R-squared is possibly the most important measurement produced by this summary. R-squared is the measurement of how much of the independent variable is explained by changes in our dependent variables. In percentage terms, 0.83 would mean our model explains 83% of the change in our ‘mpg’ variable. Adjusted (means modified to avoid have the risk of overfitting) R-squared is important for analyzing multiple independent variables’ efficacy on the model. Linear regression has the quality that your model’s R-squared value will never go down with additional variables, only equal or higher. Therefore, your model could look more accurate with multiple variables even if they are poorly contributing. The adjusted R-squared penalizes the R-squared formula based on the number of variables, therefore a lower adjusted score may be telling you some variables are not contributing to your model’s R-squared properly.

# 模型整體顯著性檢定
# Null hypothesis H0: All regression cofficients in the model are zeros 虛無假說H0: 模型中所有迴歸係數均為零
# Alternative hypothesis H1: At leat one of the cofficients is not zero. (Attain the overall model significance) 對立假說H1: 模型中至少有一迴歸係數不為零(建模才有意義)
# The F-statistic in linear regression is comparing your produced linear model for your variables against a model that replaces your variables’ effect to 0 (mpg ~ cyl + wt versus mpg ~ 1), to find out if your group of variables are statistically significant. To interpret this number correctly, using a chosen alpha value and an F-table is necessary. Prob (F-Statistic) uses this number to tell you the accuracy of the null hypothesis, or whether it is accurate that your variables’ effect is 0. In this case, it is telling us 6.81e-12 chance of this.

# Log-likelihood is a numerical signifier (summary) of the likelihood that your produced model produced the given data. It is used to compare coefficient values for each variable in the process of creating the model. AIC (Akaike information criteria) and BIC (Bayesian information criteria) are both used to compare the efficacy of models in the process of linear regression, using a penalty system for measuring multiple variables (they will punish models with too many variables !). These numbers are used for feature selection of variables.

# R^2 = 0.83代表模型解釋了83%的反應變量變異數(整體模型顯著性指標)，解釋變量cyl與wt係數均為負值且顯著(各個迴歸項顯著性指標)。

#### The second part about coefficients 第二部份是迴歸係報表
# Our first informative column is the coefficient. For our intercept, it is the value of the intercept. For each variable, it is the measurement of how change in that variable affects (cyl and wt) the dependent variable (mpg). It is the ‘m’ in ‘y = mx + b’ One unit of change in the dependent variable will affect the variable’s coefficient’s worth of change in the independent variable. If the coefficient is negative, they have an inverse relationship. As one rises, the other falls.

# Our std error is an estimate of the standard deviation of the coefficient, a measurement of the amount of variation in the coefficient throughout its data points. (Smaller is better) The t is related and is a measurement of the precision with which the coefficient was measured. A low std error compared to a high coefficient produces a high t statistic, which signifies a high significance for your coefficient.

# Null hypothesis H0: beta_{i} = 0, i = 0 (intercept), 1(cyl), 2(wt) 虛無假說H0: 迴歸係數i為零
# Alternative hypothesis H1: beta_{i} ~= (significant) 0, i = 0 (intercept), 1(cyl), 2(wt) 對立假說H1: 迴歸係數i『不』為零
# P>|t| is one of the most important statistics in the summary. It uses the t statistic to produce the p value, a measurement of how likely your coefficient is measured through our model by chance. The p value of 0.001 for cyl is saying there is a 0.001 chance the cyl variable has no affect on the dependent variable (quite small), mpg, and our results are produced by chance. Proper model analysis will compare the p value to a previously established alpha value (**specified significance level**), or a threshold with which we can apply significance to our coefficient. A common alpha is 0.05, which few of our variables pass in this instance.

# [0.025 and 0.975] are both measurements of values of our coefficients within 95% of our data, or within two standard deviations. Outside of these values can generally be considered outliers.

#### The third part on residuals 第三部份是殘差報表
# e_{i} = y_{i} - \hat{y}_{i}, i = 1, 2, ..., 32
# Omnibus describes the normalcy (normality, it means residuals follow normal/Gaussian distribution) of the distribution of our residuals using skew and kurtosis as measurements. A 0 would indicate perfect normalcy. Prob(Omnibus), p-value again, is a statistical test measuring the probability the residuals are normally distributed. A 1 would indicate perfectly normal distribution. Ho: Normal; H1: Not Normal. 殘差常態性檢驗 Skew is a measurement of symmetry in our data, with 0 being perfect symmetry. Kurtosis measures the peakiness of our data, or its concentration around 0 in a normal curve. Higher kurtosis implies fewer outliers. 偏態係數(越接近0越對稱)與峰度係數(值越高表離群值越少)的計算

# Homoscedasticity (constant variance) versus Heteroscedasticity (NOT constant variance)
# Durbin-Watson is a measurement of homoscedasticity (are the variance of residuals constant across different values), or an even distribution of errors throughout our data. Heteroscedasticity would imply an uneven distribution, for example as the data point grows higher the relative error grows higher. Ideal homoscedasticity will lie between 1 and 2. 變異數/方差齊質性檢定

# Jarque-Bera (JB) and Prob(JB) are alternate methods of measuring the same value as Omnibus and Prob(Omnibus) using skewness and kurtosis. We use these values to confirm each other. Ho: Normal; H1: Not Normal. 運用偏態與峰度係數檢定殘差的常態性

# Condition number is a measurement of the sensitivity of our model as compared to the size of changes in the data it is analyzing. Multicollinearity is strongly implied by a high condition number. 條件數值越高，暗示自變量之間有複共線性 Multicollinearity a term to describe two or more independent variables that are **strongly related** to each other and are falsely affecting our predicted variable by redundancy. 自變量之間的複共線性

#### Use residual diagnostic plots to check the assumptions under OLS 資料服從OLS模型的假設嗎？我們試著用殘差繪圖檢視之！
#### Plot 1. 殘差vs.配適值散佈圖(Residuals vs. Fitted Plot)
# Take out residuals and fitted values from the fitted model, and calculate the lowess smoothing curve 首先從配適好的模型中取出殘差值e_i與配適值\hat{y}_i，並用這些點計算lowess平滑曲線(紅色)。
# Remember to mark three samples with largest residuals 在圖中標註殘差絕對值最大的三個樣本(三款車)。

dir(results) # 有看到殘差'resid', 配適值'fittedvalues'
residuals = results.resid # e_i = y_i - \hat{y}_i
fitted = results.fittedvalues # \hat{y}_i

# Formula e_i = y_i - \hat{y}_i checking 驗算
residuals == (mtcars.mpg - fitted) # 全部為True
np.sum(residuals == (mtcars.mpg - fitted)) # 32 is # of samples

smoothed = lowess(residuals,fitted) # 無母數平滑方法, (32, 2) 紅線座標計算
top3 = abs(residuals).sort_values(ascending = False)[:3] # mark the top three residuals (absolute values), top3 is a pandas Series.

plt.rcParams.update({'font.size': 16})
plt.rcParams["figure.figsize"] = (8,7)
fig, ax = plt.subplots()
ax.scatter(fitted, residuals, edgecolors = 'k', facecolors = 'none') # scatterploy of residuals against fitted values 殘差對配適值的散佈圖, edgecolors: The edge color of the marker ('k' as black). facecolors: Set the facecolor of the Axes.
ax.plot(smoothed[:,0],smoothed[:,1],color = 'r') # smooth local regression 'red' curve 繪製紅色的局域迴歸線
ax.set_ylabel('Residuals') # low-level plotting
ax.set_xlabel('Fitted Values')
ax.set_title('Residuals vs. Fitted')
ax.plot([min(fitted),max(fitted)],[0,0],color = 'k',linestyle = ':', alpha = .3) # draw a horizontal zero line 繪製水平虛線
# Mark the top three residuals 迴圈標出三個殘差特大值
for i in top3.index:
    print(i) # Toyota Corolla, Fiat 128, Toyota Corona
    ax.annotate(i,xy=(fitted[i],residuals[i]))

plt.show()
# Results show that there is possibly nonlinear structure in the residuals, and Fiat 128, Toyota Corolla and Toyota Corona deserve more explorations. 結果顯示殘差中可能有非線性結構，Fiat 128, Toyota Corolla和Toyota Corona可能是離群資料點，值得進一步探討。


#### Plot 2. 殘差常態分位數圖(Residuals Normal Probability Plot)，另有機率-機率繪圖
# Get the sorted Studentized residuals and use stats.probplot() to have theoretical quantiles 輸入排序後的學生化殘差，stats.probplot()可獲取理論分位數值(theoretical quantiles)。
# 繪製學生化殘差與理論分位數值的散佈圖，在圖中加上45度斜直線，並註記學生化殘差絕對值最大的三個樣本。 
sorted_student_residuals = pd.Series(results.get_influence().resid_studentized_internal) # index為流水號
sorted_student_residuals.index = results.resid.index # index更換為車廠與車型，=residuals.index亦可
sorted_student_residuals = sorted_student_residuals.sort_values(ascending = True) # empirical distribution要排序後再與理論分佈的百分位數繪圖
df = pd.DataFrame(sorted_student_residuals)
df.columns = ['sorted_student_residuals']
df['theoretical_quantiles'] = stats.probplot(df['sorted_student_residuals'], dist = 'norm', fit = False)[0] # Why [0]? 因為[1]是排序後的student_residuals
rankings = abs(df['sorted_student_residuals']).sort_values(ascending = False)
top3 = rankings[:3]

fig, ax = plt.subplots()
x = df['theoretical_quantiles']
y = df['sorted_student_residuals']
ax.scatter(x,y, edgecolor = 'k',facecolor = 'none')
ax.set_title('Normal Q-Q')
ax.set_ylabel('Standardized Residuals')
ax.set_xlabel('Theoretical Quantiles')
ax.plot([np.min([x,y]),np.max([x,y])],[np.min([x,y]),np.max([x,y])], color = 'r', ls = '--')
for val in top3.index:
    ax.annotate(val,xy=(df['theoretical_quantiles'].loc[val],df['sorted_student_residuals'].loc[val]))
plt.show()

# This plot shows that residuals may follow Gaussian distribution. 圖形顯示殘差可能服從常態分佈，下一步使用尺度-位置圖(scale-location plot)，檢驗迴歸模型的變異數同/齊質性(homoskedasticity)假設。

#### Plot 3. 尺度-位置圖(scale-location plot)
# Get the sorted Studentized residuals and take absolute and square root of it. 從配適好的模型中取出標準化殘差，接著取絕對值再開方根。然後將轉換後的殘差與配適值畫散佈圖
# Check the scatterplot pattern to see if there is any violation on Homoscedasticity assumption. 如果橫跨配適值的整個範圍中，轉換後的殘差型態不一致時，則不符合變異數同質性的假設
# Also mark samples with larger residuals 一樣可以標註較大之轉換殘差樣本

student_residuals = results.get_influence().resid_studentized_internal
sqrt_student_residuals = pd.Series(np.sqrt(np.abs(student_residuals)))
sqrt_student_residuals.index = results.resid.index
smoothed = lowess(sqrt_student_residuals,fitted)
top3 = abs(sqrt_student_residuals).sort_values(ascending = False)[:3]

fig, ax = plt.subplots()
ax.scatter(fitted, sqrt_student_residuals, edgecolors = 'k', facecolors = 'none')
ax.plot(smoothed[:,0],smoothed[:,1],color = 'r')
ax.set_ylabel('$\sqrt{|Studentized \ Residuals|}$') # '\ ' means white space
ax.set_xlabel('Fitted Values')
ax.set_title('Scale-Location')
ax.set_ylim(0,max(sqrt_student_residuals)+0.1)
for i in top3.index:
    ax.annotate(i,xy=(fitted[i],sqrt_student_residuals[i]))
plt.show()

# The smooth line is increasing upward, especially if Chrysler Imperial is removed. So the Homoscedasticity assumption has been violated. 此例中似乎lowess平滑曲線有向上的趨勢，如果我們將Chrysler Imperial移除狀況可能更嚴重，因此可說此模型違背了同質性假說(此圖判斷可能較第一個圖更準)

#### Plot 4. 殘差對槓桿圖(residuals vs. leverage plot)
# Get the sorted Studentized residuals again. Leverage values which are taken from the hat matrix are on x-axis. y軸仍然是標準化殘差(或是statsmodels中的內部學生化殘差)，x軸為槓桿值(leverage)，後者從OLS的帽子矩陣(hat matrix)中的對角線元素取出
# 此圖還標示出個樣本的庫克距離(Cook’s Distance)
# 一樣標註出三個最大之取絕對值的標準化殘差樣本

student_residuals = pd.Series(results.get_influence().resid_studentized_internal)
student_residuals.index = results.resid.index
df = pd.DataFrame(student_residuals)
df.columns = ['student_residuals']
# New thing coming up !
df['leverage'] = results.get_influence().hat_matrix_diag
smoothed = lowess(df['student_residuals'],df['leverage'])
sorted_student_residuals = abs(df['student_residuals']).sort_values(ascending = False)
top3 = sorted_student_residuals[:3]

fig, ax = plt.subplots()
x = df['leverage']
y = df['student_residuals']
xpos = max(x)+max(x)*0.01  
ax.scatter(x, y, edgecolors = 'k', facecolors = 'none')
ax.plot(smoothed[:,0],smoothed[:,1],color = 'r')
ax.set_ylabel('Studentized Residuals')
ax.set_xlabel('Leverage')
ax.set_title('Residuals vs. Leverage')
ax.set_ylim(min(y)-min(y)*0.15,max(y)+max(y)*0.15)
ax.set_xlim(-0.01,max(x)+max(x)*0.05)
plt.tight_layout()
for val in top3.index:
    ax.annotate(val,xy=(x.loc[val],y.loc[val]))

cooksx = np.linspace(min(x), xpos, 50)
p = len(results.params)
poscooks1y = np.sqrt((p*(1-cooksx))/cooksx)
poscooks05y = np.sqrt(0.5*(p*(1-cooksx))/cooksx)
negcooks1y = -np.sqrt((p*(1-cooksx))/cooksx)
negcooks05y = -np.sqrt(0.5*(p*(1-cooksx))/cooksx)

ax.plot(cooksx,poscooks1y,label = "Cook's Distance", ls = ':', color = 'r')
ax.plot(cooksx,poscooks05y, ls = ':', color = 'r')
ax.plot(cooksx,negcooks1y, ls = ':', color = 'r')
ax.plot(cooksx,negcooks05y, ls = ':', color = 'r')
ax.plot([0,0],ax.get_ylim(), ls=":", alpha = .3, color = 'k')
ax.plot(ax.get_xlim(), [0,0], ls=":", alpha = .3, color = 'k')
ax.annotate('1.0', xy = (xpos, poscooks1y[-1]), color = 'r')
ax.annotate('0.5', xy = (xpos, poscooks05y[-1]), color = 'r')
ax.annotate('1.0', xy = (xpos, negcooks1y[-1]), color = 'r')
ax.annotate('0.5', xy = (xpos, negcooks05y[-1]), color = 'r')
ax.legend()
plt.show()

# Same observation as above, Chrysler Imperial is an influential observation ! 與前圖的發現一致，Chrysler Imperial對於模型有比較大的影響力，因為它是配適值可能範圍中最小邊緣的離群樣本

# Remember there are six residual diagnostic plots in R function plot.lm(). 其實在R語言中，提供了六種殘差診斷圖

#### Reference: Going from R to Python — Linear Regression Diagnostic Plots(https://towardsdatascience.com/going-from-r-to-python-linear-regression-diagnostic-plots-144d1c4aa5a)

#### 案例二：資料理解與整理
#from statsmodels.compat import lzip
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
#from statsmodels.formula.api import ols
import pandas as pd

df = pd.read_csv('./Air_106鳳山.csv')
df.drop(df.columns[0], axis=1, inplace=True)

df.isnull().sum()

# 垂直刪除遺缺過多的變量
df.drop(['PH_RAIN', 'RAIN_COND', 'RAINFALL'], axis=1, inplace=True)

tmp = df.isnull().sum(axis='columns')
# 接著橫向刪除不完整的樣本
df.dropna(axis=0, inplace=True)

df.head(3)

df.columns
# Pollutant Standards Index, PSI
# CH4: 甲烷
# CO: 一氧化碳
# NHMC: 非甲烷總烴
# NO: 一氧化氮
# NO2: 二氧化氮
# NOx: 氮氧化物
# O3: 臭氧
# PM10:
# PM2.5:
# RH:
# SO2: 二氧化硫
# THC: 總碳氫有機氣體
# WD_HR:
# WIND_DIREC:
# WIND_SPEED:
# WS_HR:
# FPMI: 細懸浮微粒指標

# opendata.epa.gov.tw 空氣品質即時污染指標 (https://sheethub.com/opendata.epa.gov.tw/空氣品質即時污染指標)
# 環境保護標準中的VOCs、TVOCs、TVOC、NMHC、HC 都是啥？(https://kknews.cc/other/yzql3ob.html)
# 空氣污染懶人包(http://nehrc.nhri.org.tw/toxic/ref/懶人包_v3-rev2.pdf)

df.dtypes

# 已無遺缺值
df.isnull().sum()

aqCorr = df.corr()

import seaborn as sns
sns.set(style="darkgrid")

# Generate a mask for the upper triangle
mask = np.zeros_like(aqCorr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize = (9, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True) # https://zhuanlan.zhihu.com/p/27471537
sns.heatmap(aqCorr, mask=mask, annot=True, cmap=cmap, ax=ax)
f.tight_layout()

# 二氧化氮與溫度的相關係數
import scipy

PKP = scipy.stats.pearsonr(df['NO2'], df['AMB_TEMP'])
PKP = np.round(PKP, decimals=4)
print("Association between NO2 and temperature is {}".format(PKP))

# 二氧化氮與一氧化碳的相關係數
PKP = scipy.stats.pearsonr(df['NO2'], df['CO'])
PKP = np.round(PKP, decimals=4)
print("Association between NO2 and CO is {}".format(PKP))

# 自變數矩陣與反應變數向量
X = df[['AMB_TEMP','CO']]
y = df['NO2']

# 加入常數項後配適模型
model = sm.OLS(y, sm.add_constant(X))
model_fit = model.fit()

type(model_fit) # statsmodels.regression.linear_model.RegressionResultsWrapper
dir(model_fit) # 有看到'summary'、'fittedvalues'、'resid'、'get_influence'

print(model_fit.summary())

#為了後續的繪圖，合併自變數矩陣與反應變數向量
dfMy = pd.concat([X, y], axis=1) # (8344, 3)

#### 1. 繪製殘差對配適值散佈圖(Residuals vs Fitted plot)，以瞭解線性模型是否妥善捕捉到反應變數的變異性

# 反應變數預測值
model_y = model_fit.fittedvalues

# 繪圖前參數設定
plt.style.use('seaborn')

plt.rc('font', size=14)
plt.rc('figure', titlesize=18)
plt.rc('axes', labelsize=15)
plt.rc('axes', titlesize=18)

pl_A = plt.figure()
pl_A.axes[0] = sns.residplot(model_y, # Data or column name in data for the predictor variable.
 dfMy.columns[-1], data=dfMy, # Data or column name in data for the response variable.
                          lowess=True,
                          scatter_kws={'color': 'grey','alpha': 0.4},
                          line_kws={'color': 'yellow', 'lw': 1, 'alpha': 1.0})
pl_A.axes[0].set_title('Residuals vs Fitted')
pl_A.axes[0].set_xlabel('Fitted values')
pl_A.axes[0].set_ylabel('Residuals')

# 在OLS線性模型的假設下，lowess配適的曲線應該要相當接近水平虛線(代表隨機噪訊)

# 黃色弧形線顯示模型並未捕捉到某些非線性的成分，或許指數模型或以其他非線性轉換後的人工變數來建模會更好

#### 2. 常態分位數圖 Normal Q-Q Plot
from statsmodels.graphics.gofplots import ProbPlot # 機率繪圖

#### 殘差種類
# 普通殘差
# 標準化殘差(變異數均為1)
# 學生化殘差(可用來測試離群值)
# Pearson's殘差(加權最小平方法WLS)

# 各樣本普通殘差與標準化殘差
# get_influence()方法計算樣本影響值與離群程度衡量
model_residuals = model_fit.resid
model_norm_residuals = model_fit.get_influence().resid_studentized_internal # 有正有負

QQ = ProbPlot(model_norm_residuals) # 輸入A 1d data array
dir(QQ)

pl_B = QQ.qqplot(line='45', alpha=0.5, color='grey', lw=1)
pl_B.axes[0].set_title('Normal Q-Q')
#pl_B.axes[0].set_xlabel('Theoretical Quantiles')
pl_B.axes[0].set_ylabel('Standardized Residuals');
# 標示出殘差絕對值最大的三個樣本
abs_norm_resid = np.flip(np.argsort(np.abs(model_norm_residuals)), 0) # np.argsort: returns the indices that would sort an array; reverse the order of elements in an array along the given axis.
abs_norm_resid_top_3 = abs_norm_resid[:3]
for r, i in enumerate(abs_norm_resid_top_3):
    pl_B.axes[0].annotate(i,
                               xy=(np.flip(QQ.theoretical_quantiles, 0)[r],
                                   model_norm_residuals[i]))

# 上圖顯示標準化殘差並非完美地服從常態分佈(圖形顯示標準化殘差為厚尾heavy tails分佈)
# 如果標準化殘差不服從常態分佈，小心模型可能產生極端結果

# Let’s check which data load normal distribution
dfMy.iloc[6615:6665, :].plot(kind='bar',figsize=(16,2))

# Clearly, observation 6970, indicated by the Normal Q-Q Plot chart, significantly differs from the other data. To improve the quality of the regression model, outliers should be discarded from the data.

#### 3. 尺度位置圖(Scale-Location plot)
# 橫軸為配適值，縱軸為標準化殘差取絕對值再開方根
# 此圖檢核變異數的異質性(heteroscedasticity)，線性迴歸模型變異數齊質性的假設是y的每一個機率分佈都有相同的標準差，無論預測變數x的值為何

model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals)) # 標準化殘差取絕對值再開方根
model_abs_resid = np.abs(model_residuals) # 普通殘差取絕對值

pl_F = plt.figure()
plt.scatter(model_y, model_norm_residuals_abs_sqrt, color='grey', alpha=0.5);
sns.regplot(model_y, model_norm_residuals_abs_sqrt,
              scatter=False,
              ci=False,
              lowess=True,
              line_kws={'color': 'yellow', 'lw': 1, 'alpha': 1.0});
pl_F.axes[0].set_title('Scale-Location')
pl_F.axes[0].set_xlabel('Fitted values')
pl_F.axes[0].set_ylabel('$sqrt{|Standardized Residuals|}$')

# lowess配適的黃色曲線，如果越接近水平線，則資料的變異數越有可能是齊質的(homoscedastic)
# 尺度與位置的異質偵測圖可能呈現V字形，這代表兩側值較中間為高，顯示模型並未揭露資料中的非線性結構，因此需要進一步調查非線性模型。

# The two most common methods of “consolidating” heteroscedasticity are:

# use of the least squares weighing method or
# use of heteroscedastically corrected covariance matrix (hccm).
# The next graph indicated observation number 1646 as a segment disrupting the linear regression model.

#### 4. 殘差對槓桿圖(Residuals vs Leverage plot)
# When standardized residues do not have a normal distribution, extreme values of y results may occur. In the case of high leverage points, extremely independent x variables may appear. Extreme x seems to be so bad, but may have a detrimental effect on the model because the coefficients at x or β are very sensitive to leverage points. The purpose of the Residuals vs Leverage chart is to identify these problematic observations.

model_leverage = model_fit.get_influence().hat_matrix_diag # 從hat_matrtix的對角線取出槓桿值
model_cooks = model_fit.get_influence().cooks_distance[0] # 取出庫克距離

pl_C = plt.figure(figsize=(16,4));
plt.scatter(model_leverage, model_norm_residuals, color='grey', alpha=0.3);
sns.regplot(model_leverage, model_norm_residuals,
              scatter=False,
              ci=False,
              lowess=True,
              line_kws={'color': 'blue', 'lw': 1, 'alpha': 1.0});
pl_C.axes[0].set_xlim(0, max(model_leverage)+0.01)
pl_C.axes[0].set_ylim(-3, 5)
pl_C.axes[0].set_title('Residuals vs Leverage')
pl_C.axes[0].set_xlabel('Leverage')
pl_C.axes[0].set_ylabel('Standardized Residuals');


leverage_top_3 = np.flip(np.argsort(model_cooks), 0)[:3]
for i in leverage_top_3:
    pl_C.axes[0].annotate(i,
                                xy=(model_leverage[i],
                                    model_norm_residuals[i]));

# Thanks to the Cook distance, we only need to find leverage points that have a distance greater than 0.5. These 0.5 are shown on the graph as a dashed curve at the top and bottom – shown when such outliers occur. We do not have any leverage points in this chart that are outside the 0.5 curve. Therefore, there are no outliers in the top right or bottom right of the chart.
# The procedure with the Residuals vs Leverage chart is that outliers are removed from independent variables and the model is rebuilt. This procedure improves model properties.

# The chart showed observation number 5690. Let’s see what the data is.
    
dfMy.iloc[5660:5710, :].plot(kind='bar',figsize=(16,2), legend=False)

#### 5. 影響圖(Influence plot)
# The impact graph shows the residual values of the model as a function of the lever of each observation measured with the hat matrix. Externally learned residual values are scaled according to their standard deviation
# Two impact measures are available: Cook and DFFITS.

fig, ax = plt.subplots(figsize=(18,3))
fig = sm.graphics.influence_plot(model_fit, ax=ax, criterion="cooks")

# Interpretation

# The size of the bubbles (in our chart you can not see it) means the size of the cook distance, the larger the bubble, the greater the cook parameter.


# DFFITS¶

# DFFITS is a diagnostic that is intended to show how much impact a point in the statistical regression proposed in 1980 has [1] It is defined as student DFFIT, where the latter is a change in the predicted value for the point obtained when this point is left outside the regression .

# Cooks¶

# is a commonly used estimation of the data point impact when performing the least squares regression analysis. [1] In practical ordinary least squares analysis, the Cook distance can be used in several ways:

# to identify influential data points that are particularly worth checking for validity; or
# indicate areas of the design space in which it would be good to obtain more data points.
# This means that Cooks distance measures the impact of each observation in the model or “what would happen if each observation were not in the model,” and this is important because it is one way to detect outliers that particularly affects the regression line. When we are not looking for and treating potential outliers in our data, it is possible that the corrected coefficients for the model may not be the most representative or appropriate, which may lead to incorrect inferences.

# The hat values¶

# Hat values ​​are fitted values ​​or predictions made by the model for each observation. Completely different from Cook’s distance.

# H levarage¶

# H levarage measures how each input parameter X affects the fit model. In contrast, the Cook distance also includes the influence of the output parameter y.

#### R語言殘差診斷圖共有六種

#### References:
# LINEAR REGRESSION DIAGNOSTICS part 1.(http://sigmaquality.pl/uncategorized/air-pollution-analysis_an-example-of-the-application-of-linear-regression-diagnostics-in-statsmodels-301220191250/)
# Emulating R plots in Python (https://emredjan.github.io/blog/2017/07/11/emulating-r-plots-in-python/)
# https://medium.com/swlh/interpreting-linear-regression-through-statsmodels-summary-4796d359035a









