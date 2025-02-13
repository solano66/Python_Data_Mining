'''
Collated by Prof. Ching-Shih Tsou 邹庆士 教授 (Ph.D.) at the IDS, CICD, and CISD of NTUB (台北商业大学资讯与决策科学研究所暨智能控制与决策研究室教授兼校务永续发展中心主任); at the ME Dept. and CAIDS of MCUT (2020年借调至明志科技大学机械工程系担任特聘教授兼人工智慧暨资料科学研究中心主任两年); the CSQ (2019年起任品质学会大数据品质应用委员会主任委员); the DSBA (2013年创立台湾资料科学与商业应用协会); and the CARS (2012年创立中华R软体学会)
Notes: This code is provided without warranty.
'''

#### 2.1.5 R语言资料清理
import numpy as np
x = [1, 2, 3, np.NaN]
# 向量元素加总产生nan
y = x[0] + x[1] + x[2] + x[3]
y

# 加总函数的结果也是NA
z = np.sum(x)
z

# 移除NA后再做加总计算
z = np.nansum(x)
z

# pandas Series遗缺值NA辨识函数
import pandas as pd
pd.Series(x).isnull() # Similar to is.na() in R

# 取得遗缺值位置/样本编号(Which one is TRUE?)
np.where(pd.Series(x).isnull()) # Similar to which() in R

#### Data cleaning on missing values
import pandas as pd
from io import StringIO

csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,,,8.0
0.0,11.0,12.0,'''

csv_data
type(csv_data) # str

StringIO(csv_data)
type(StringIO(csv_data)) # _io.StringIO

df = pd.read_csv(StringIO(csv_data))
df

help(df.isnull)

df.isnull()
df.isnull().sum() # from variable perspective
df.values # np.array behind

df.dropna()
df.dropna(axis=1)
df.dropna(how='all') # default 'any'

df.dropna(thresh=4) # drop rows that have not at least 4 non-NaN values

df.dropna(subset=['C']) # drop rows that column 'C' has NaN

#### Imputation by scikit-learn
import sklearn
sklearn.__version__ # '0.23.1' (from v0.20.4 to v0.22.2 is the grace period)
#from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer
#imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr = imr.fit(df)
imputed_data = imr.transform(df.values) # input numpy.ndarray
imputed_data

imputed_data = imr.transform(df) # you can also input pandas.DataFrame
imputed_data # same result

#### Case study: algae data set (page 164)
#### 2.1.6 Python语言资料清理
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
algae = pd.read_csv("algae.csv")

pd.set_option('display.max_row', 500)
pd.set_option('display.max_column', 500)

#### 各变量摘要统计表整合在资料表下方
algae_summary = algae.describe(include='all')
algae_new = pd.concat([algae, algae_summary])

#### 单变量遗缺值检查
# R 语言语法可想成是head(isnull(algae['mxPH'])))
print(algae['mxPH'].isnull().head())

nr_nan = algae['mxPH'].isnull().sum()

# 注意Python 输出格式化语法({} 搭配format() 方法)
print(" 遗缺 {} 笔观测值".format(nr_nan))

## ------------------------------------------------------------------------
# 利用pandas 序列方法dropna() 移除单变量遗缺值
mxPH_naomit = algae['mxPH'].dropna()
print(len(mxPH_naomit))

#### 检视整个资料表的遗缺状况
print(algae.isnull().iloc[45:55,:5])
# 横向移除不完整的观测值(200 笔移除16 笔)
algae_naomit = algae.dropna(axis=0)
print(algae_naomit.shape)

#### 移除遗缺程度严重的样本
# 以thresh 引数设定最低变数个数门槛(200 笔移除2 笔: 61和198)
algae_over13 = algae.dropna(thresh=13)
print(algae_over13.shape)

#### 纵横统计遗缺样貌
# 各变数遗缺状况：Chla 遗缺观测值数量最多, Cl 次之...
algae_nac = algae.isnull().sum(axis=0)
print(algae_nac)
# 各观测值遗缺状况：遗缺变数个数
algae_nar = algae.isnull().sum(axis=1)
print(algae_nar[60:65])

#### 检视不完整的观测值
# algae_nar>0 回传横向遗缺数量大于0 的样本
print(algae[algae_nar > 0])
# 遗缺变数个数大于0(i.e. 不完整) 的观测值编号
print(algae[algae_nar > 0].index)

# 检视遗缺变数超过变数个数algae.shape[1] 之20% 的观测值
print(algae[algae_nar > algae.shape[1]*.2])
# 如何获取上表的横向索引值？
print(algae[algae_nar > algae.shape[1]*.2].index)

## ------------------------------------------------------------------------
# 以drop() 方法，给IndexRange，横向移除遗缺严重的观测值
algae=algae.drop(algae[algae_nar > algae.shape[1]*.2].index)
print(algae.shape)

#### 以下为2.1.6 Python语言资料清理补充代码(填补方式)
#### mxPH单一补值
mxPH = algae['mxPH'].dropna() # 须先移除NaNs后再绘图或计算
#fig, ax = plt.subplots()
#ax.hist(mxPH, alpha=0.9, color='blue')
#plt.show() # 近乎对称钟型分布

#ax = plt.gca() # 绘图的多种方法
## the histogram of the data
#ax.hist(mxPH, bins=35, color='r')
#ax.set_xlabel('Values')
#ax.set_ylabel('Frequency')
#ax.set_title('Histogram of mxPH')
#plt.show()

fig = plt.figure() # 绘图的多种方法
ax = fig.add_subplot(111) # 图面布局之 2,1,1 or 2,1,2 行、列、图
ax.hist(mxPH) # high-level plotting 高阶绘图
ax.set_xlabel('mxPH Values') # low-level plotting 低阶绘图的画龙点睛
ax.set_ylabel('Frequency') # low-level plotting
ax.set_title('Histogram of mxPH') # low-level plotting
plt.show()
#### 总结：图面宣告与布局、数据与高阶绘图、低阶绘图的画龙点睛

#### 常态机率绘图或分位数图
import statsmodels.api as sm
from matplotlib import pyplot as plt
fig = sm.qqplot(mxPH) # 高阶绘图函数qqplot from statsmodels.api
plt.show()

#### 算术平均数单一补值
print(algae['mxPH'].describe())
mean = algae['mxPH'].mean() # pandas Series自动排除nan后计算mean，此举与Python numpy套件做法不同！水很深 ~
algae['mxPH'].fillna(mean, inplace=True) # 以算术平均数填补唯一的遗缺值
print(algae['mxPH'].describe()) # 确认是否填补完成

#### Chla单一补值
Chla = algae['Chla'].dropna() # 须先移除NaNs后再绘图或计算
fig, ax = plt.subplots() # 有发现111被省略了吗！人性本懒
ax.hist(Chla, alpha=0.9, color='magenta')
ax.set_xlabel('Chla Values')
ax.set_ylabel('Frequency')
ax.set_title('Histogram of Chla')
plt.show() # 右偏不对称分布

#### 常态机率绘图或分位数图
import statsmodels.api as sm
from matplotlib import pyplot as plt
fig = sm.qqplot(Chla)
plt.show()

print(algae['Chla'].describe()) # count 188, 12 missings
median = algae['Chla'].median() # pandas Series自动排除nan后计算median，此举与Python numpy套件做法不同！水很深 ~
algae['Chla'].fillna(median, inplace=True) # 以中位数(50%分位数)填补遗缺值
print(algae['Chla'].describe()) # 确认是否填补完成(count 198)

#### 多变量补值
alCorr = algae.corr() # 自动挑数值变数计算相关系数 r -> correlation coefficient matrix，PO4与oPO4高相关(-1 <= r <= 1)

alCorr
alCorr.shape

#### 相关系数方阵视觉化
import numpy as np
# Python好用的多变量绘图套件
import seaborn as sns
# 仍需搭配基础绘图套件matplotlib
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
# 建立上三角遮罩矩阵
mask = np.zeros_like(alCorr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# 图面与调色盘设定(https://zhuanlan.zhihu.com/p/27471537)
f, ax = plt.subplots(figsize = (9, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True) 
# 绘制相关系数方阵热图
sns.heatmap(alCorr, mask=mask, annot=True, cmap=cmap, ax=ax)
f.tight_layout()


# 绘制另一种形式的相关系数方阵热图
sns.clustermap(alCorr, annot=True, cmap=cmap)
f.tight_layout()

#### 相关系数符号方阵
#for i in range(15):
#  for j in range(15):
#    # 如果完全正相关或完全负相关的则不需要更改数字
#    if alCorr.iloc[i,j] == -1 or alCorr.iloc[i,j] == 1:break
#    elif alCorr.iloc[i,j] < 0.3:alCorr.iloc[i,j]=' '
#    elif alCorr.iloc[i,j] >= 0.3 and alCorr.iloc[i,j] < 0.6:
#      alCorr.iloc[i,j]='.'
#    elif alCorr.iloc[i,j] >= 0.6 and alCorr.iloc[i,j] < 0.8:
#      alCorr.iloc[i,j]=','
#    elif alCorr.iloc[i,j] >= 0.8 and alCorr.iloc[i,j] < 0.9:
#      alCorr.iloc[i,j]='+'
#    elif alCorr.iloc[i,j] >= 0.9 and alCorr.iloc[i,j] < 0.95:
#      alCorr.iloc[i,j]='*'
#    elif alCorr.iloc[i,j] >= 0.95 and alCorr.iloc[i,j] < 1:
#      alCorr.iloc[i,j]='B'
#    
#    # 对角线以后的值删除使矩阵变成下三角矩阵
#    for k in range((i+1),15):alCorr.iloc[i,k]=' '
#  
#print(alCorr)

#### PO4多变量补值
# https://github.com/statsmodels/statsmodels/issues/5343
# !pip install --upgrade patsy
import statsmodels.formula.api as sm
result = sm.ols(formula="PO4 ~ oPO4", data=algae).fit() # ols: ordinary least square

type(result) # statsmodels.regression.linear_model.RegressionResultsWrapper
# 查询statsmodels下RegressionResultsWrapper类物件下属性及方法
# 初学方式
dir(result)

# 进化的过程
# Not callable表属性
[name for name in dir(result) if not callable(getattr(result, name))]
# Callable表方法
[name for name in dir(result) if callable(getattr(result, name))]

# 最聪明的方式
[(name, type(getattr(result, name))) for name in dir(result)]

# 填补会用到的回归方程系数
print(result.params)

# statsmodels有完整的统计报表
print(result.summary())

type(result.params)

#### 运用回归方程填补遗缺值 
algae.at[27, 'PO4'] = result.params[0] + result.params[1]*algae.loc[27]['oPO4'] # pandas.DataFrame改值要用set_value(列编号, 行编号或名称, 补入之值) 0.21.0 deprecated
algae.loc[27]['PO4']

result.params[0] + result.params[1]*algae.loc[27]['oPO4']

algae = pd.read_csv("./algae.csv")
algae = algae.dropna(thresh=13)

# 创造多个PO4遗缺值的情境
algae.PO4[28:33]=np.nan # Warning!

# 考虑连自变数oPO4都遗缺的边界案例(edge case)(参见3.5节)
algae.oPO4[32]=np.nan

algae_nar = algae.PO4.isnull()
print(algae[algae_nar == True][['oPO4', 'PO4']])

def fillPO4(oP):
    # 边界案例判断与处理
    if np.isnan(oP): return np.nan
    # 否则，运用模型物件result中回归系数进行补值计算
    else: return result.params[0] + result.params[1]*oP

# 逻辑值索引、隐式回圈与自订函数
algae.PO4[np.isnan(algae.PO4)==True] = algae.oPO4[np.isnan(algae.PO4)==True].apply(fillPO4)

algae.loc[27:32,['oPO4','PO4']]

