'''
Collated by Prof. Ching-Shih Tsou 邹庆士 教授 (Ph.D.) at the IDS, CICD, and CISD of NTUB (台北商业大学资讯与决策科学研究所暨智能控制与决策研究室教授兼校务永续发展中心主任); at the ME Dept. and CAIDS of MCUT (2020年借调至明志科技大学机械工程系担任特聘教授兼人工智慧暨资料科学研究中心主任两年); the CSQ (2019年起任品质学会大数据品质应用委员会主任委员); the DSBA (2013年创立台湾资料科学与商业应用协会); and the CARS (2012年创立中华R软体学会)
Notes: This code is provided without warranty.
Dataset: segmentationOriginal.csv
'''

#### Cell Segmentation Case (本节2.3.1的running example)
import pandas as pd
import numpy as np
cell = pd.read_csv('segmentationOriginal.csv')

import sys
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', sys.maxsize)

#### 0. Data Architecture Understanding and Missing Values Identifying 数据架构基本理解与缺失值辨识
cell.info() # RangeIndex, Columns, dtypes, memory usage

# 查看cell DataFrame的栏位
print(cell.columns) # pandas包下的Index对象

cell.dtypes

cell_stats = cell.describe(include = "all").T # describe: descriptive statistics 摘要/叙述统计值
# cell_stats.to_excel("cell_stats.xls") # ModuleNotFoundError: No module named 'xlwt'; !conda install xlwt --y

cell.isnull().any() # check NaN/NA by column, same as cell.isnull().sum(axis=0). No missing !

cell.isnull().values.any() # False, means no missing value ! Check the difference between above two !!!!

#cell.isnull()
#type(cell.isnull()) # pandas.core.frame.DataFrame, so .index, .column, and .values three important attributes

#cell.isnull().values
#type(cell.isnull().values) # numpy.ndarray

cell.isnull().sum() # No missing !

#### 1. Select the training set 挑出训练集
# 确认只有训练train与测试test两种样本
cell['Case'].unique()
 
# 再了解训练与测试样本的次数分布
cell.Case.value_counts() # 取单栏的句点语法
# select the training set by logical/boolean indexing 逻辑值索引
cell_train = cell.loc[cell['Case']=='Train'] # same as cell[cell['Case']=='Train'], logical indexing + broadcasting in Python or recycling in R of 'Train' + vectorization (逻辑值索引 + 短字串自动放长 + 向量元素各自比较)
# cell[[cell['Case']=='Train']] # KeyError: "None of [Index([(False, True, ...)], dtype='object')] are in the [columns]"
cell_train.head()

# 注意cell['Case']与cell[['Case']]的区别！R语言亦有类似的情况(drop = T or F)！

#### 2. Create class label vector (y) 建立类别标签向量 (label encoding 标签编码 and one-hot encoding 单热编码，类似虚拟编码)

# 类别标签向量独立为segTrainClass
segTrainClass = cell_train.Class
segTrainClass[:5]

# 标签编码
Class_label = segTrainClass.map({'PS': 1, 'WS': 2}).astype(int)

# 单热编码最快的方法 The fast way to do one-hot encoding
Class_ohe_pd = pd.get_dummies(cell['Class'])
print(Class_ohe_pd.head()) # 有栏位/变量名称

# 如何做虚拟编码(dummy coding)？
Class_dum_pd = pd.get_dummies(cell['Class'], drop_first = True) # 请留意结果只有'WS', 'PS'已被drop掉了！
print(Class_dum_pd.head()) # 有栏位/变量名称

# R语言需要吗？何时需要？何时不需要？

#### 3. Create feature matrix (X) 建立属性矩阵
cell_data = cell_train.drop(['Cell','Class','Case'], axis = 'columns')
cell_data.head()

#### 4. Differentiate categorical features from numeric features 区分类别与数值属性
# 变数名称中有"Status" versus 没有"Status"

# 法ㄧ：写回圈，最直觉！
status = [] # 名称包含'Status'变数集合
for h in range(len(cell_data.columns)):
    if "Status" in list(cell_data.columns)[h]:
        status.append(list(cell_data.columns)[h])

cell_num = cell_data.drop(status, axis=1)
cell_num.head()
# cell_num.to_csv('cell_num.csv')

not_status = [] # 名称无'Status'变数集合
for h in range(len(cell_data.columns)):
    if "Status" not in list(cell_data.columns)[h]:
        not_status.append(list(cell_data.columns)[h])

cell_cat = cell_data.drop(not_status, axis=1)
cell_cat.head()
# cell_cat.to_csv('cell_cat.csv')

# 法二： The most succinct way I think 最简洁
cell_cat = cell_data.filter(regex='Status') # Note that this routine does not filter a dataframe on its contents. The filter is applied to the labels of the index.
cell_cat.head()

# 确认是否均为类别变量
# 隐式implicit vs explicit回圈的运用
cell_cat.apply(lambda x: x.value_counts(), axis=0) # 成批产制次数分配表(important in Big Data era)

#### 5. Pick out low variance feature(s) 低变异/方差过滤
# scikit-learn (from scipy下kits for machine learning) -> sklearn (sk stands for scikit)

# Step 1 套件载入
from sklearn.feature_selection import VarianceThreshold

# Step 2 宣告空模
sel = VarianceThreshold(threshold = 0.16) # 0.16

# Step 3 & 4: 传入样本拟合实模 & 转换 fit and transform on same object
sel.fit_transform(cell_num).shape # (1009, 49), nine low variance features already removed 九个低变异变量已经被移除，到底哪九个！？
dir(sel)

sel.get_support() # 传回58个真假值
import numpy as np
unique, counts = np.unique(sel.get_support(), return_counts=True)
dict(zip(unique, counts)) # {False: 9, True: 49}

idx = sel.get_support(indices=True) # 传回留下来的49个变数编号
set(range(58))-set(idx) # 利用集合的差集运算，产生移除掉的9个变数编号

cell_num.columns[~sel.get_support()] # 逻辑值索引again！传回移除掉的9个变数名称(~ like ! in R)
cell_num.columns[list(set(range(58))-set(idx))]

#### 常问的问题
# 标准差或变异数门槛值如何决定？依各变量标准差在整个变数集的分布情况决定，没有标准答案。或者在domain中已有经验(eg. 各点位的上下限值)，则可援引此标准。The last resort ~ 与后续建模方法结合，依最终预测绩效决定合宜的门槛值！
cell_num.std().hist()

# How to decide what threshold to use for removing low-variance features? (https://datascience.stackexchange.com/questions/31453/how-to-decide-what-threshold-to-use-for-removing-low-variance-features)

#### 过度分散(percentUnique 10%)与过度集中(freqRation 95/5=19)的变数
# percentUnique为独一无二的类别值数量与样本大小的比值(10%，太高表过度分散！)
cell_cat.dtypes
percentUnique = cell_cat.AngleStatusCh1.nunique()/cell_cat.shape[0]

# freqRatio为最频繁的类别值频次，除以次频繁类别值频次的比值(95/5，太高表过度集中！)
np.unique(cell_cat.AngleStatusCh1, return_counts=True)
freq = cell_cat.AngleStatusCh1.value_counts()
freqRatio = freq[0]/freq[1]

#### 6. Transform skewed feature(s) by Box-Cox Transformation 偏斜分布属性Box-Cox 转换
# 判断变量分布是否偏斜的多种方式：1. 比较平均数与中位数; 2. 最大值与最小值的倍数，倍比大代表数值跨越多个量纲/级order of mgnitude; 3. 计算偏态系数; 4. 绘制直方图、密度曲线、盒须图等变量分布视觉化图形; 5. 检视分位数值quantiles, percentiles, quartiles
cell_num['VarIntenCh3'].describe() # 没有偏态系数，只提供*平均值*、标准差及其他*位置量数(含中位数)*
cell_num['VarIntenCh3'].max()/cell_num['VarIntenCh3'].min()
cell_num['VarIntenCh3'].skew() # 理论值域：-Inf ~ Inf, 可能的合理范围：-1 ~ 1, -2 ~ 2, -3 ~ 3 (比较夸张)
cell_num['VarIntenCh3'].hist()

# seaborn套件的displot是直方图搭配密度曲线
import seaborn as sns
sns.distplot(cell_num.VarIntenCh3) # DISTribution Plot: 直方图加上密度曲线来看分布

# 最客观的方式还是偏斜系数，所有58量化变数的偏斜系数产生一张表，降幂排列
cell_num.skew(axis=0).sort_values(ascending=False)

# python plot multiple histograms (https://stackoverflow.com/questions/47467077/python-plot-multiple-histograms)
# 取出右偏前九高的变数名称
highlyRightSkewed = cell_num.skew(axis=0).sort_values(ascending=False).head(n=9).index.values

import matplotlib.pyplot as plt
cell_num[highlyRightSkewed].hist(figsize = (30, 30))
plt.tight_layout()
plt.show()

# 取出左偏前九高的变数名称
highlyLeftSkewed = cell_num.skew(axis=0).sort_values(ascending=False).tail(n=9).index.values

cell_num[highlyLeftSkewed].hist(figsize = (30, 30))
plt.tight_layout()
plt.show()

#### Box-Cox Transformation
# 先试AreaCh1前六笔(只接受一维阵列，自动估计lambda)
from scipy import stats
print(cell['AreaCh1'].head(6))
stats.boxcox(cell['AreaCh1'].head(6))

# Separate positive predictors 挑出变量值恒正的预测变量集合
pos_indx = np.where(cell_data.apply(lambda x: np.all(x > 0)))[0]
cell_data_pos = cell_data.iloc[:, pos_indx]
cell_data_pos.head()
#help(np.all)

#### 7. Dimensionality Reduction (dr) by PCA 主成份分析维度缩减
# Step 1
from sklearn.decomposition import PCA

# Step 2
dr = PCA() # Principal Components Analysis 主成份分析，透过矩阵分解decomposition，预设会提取出min(n_samples, n_features)=58主成份，可改成PCA(n_components = 20)

# Steps 3&4
# 分数矩阵cell_pca (cell_num 旧空间 --转轴--> cell_pca 新空间)
cell_pca = dr.fit_transform(cell_num) # PCA只能针对量化变数计算
cell_pca

# 确认主成份之间是否独立无关
cor = pd.DataFrame(cell_pca).corr()
(cor > 0.0001).sum().sum() # 58，确实独立无关！

(np.corrcoef(cell_pca, rowvar=False) > 0.0001).sum() # 58

# 检视模型拟合完后，有无新增的属性与方法(通常一定有！)
dir(dr)

# 负荷矩阵或旋转矩阵
# 前十个主成份与58个原始变数的(线性组合)关系
dr.components_[:10] # [:10] can be removed.
type(dr.components_) # numpy.ndarray
dr.components_.shape # (58主成份, 58原始变量)的方阵

# 陡坡图(scree plot)决定取几个主成份
dr.explained_variance_ratio_ # 各个主成份诠释资料集总变异量的百分比(依序递减排列)
import matplotlib.pyplot as plt
plt.plot(range(1, 26), dr.explained_variance_ratio_[:25], '-o')
plt.xlabel('# of components')
plt.ylabel('ratio of variance explained')

# 可能可以降到**五维**空间中进行后续分析(Why 5D? from above scree plot 上面陡坡图)
cell_dr = cell_pca[:,:5]
cell_dr # 后续建模可考虑此PCA降维后的数据矩阵
# pd.DataFrame(cell_dr).to_csv('cell_dr.csv')

#### 8. Feature Selection by Correlation Filtering 高相关过滤
# 正确结果
corr_matrix = cell_num.corr()

import seaborn as sns
# 仍需搭配基础绘图套件matplotlib
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
# 建立上三角遮罩矩阵
mask = np.zeros_like(corr_matrix, dtype=np.bool_) # (58, 58)全为假False值的矩阵
mask[np.triu_indices_from(mask)] = True # 上三角(triu: upper triangle)遮盖(mask)起来
# 图面与调色盘设定(https://zhuanlan.zhihu.com/p/27471537)
f, ax = plt.subplots(figsize = (36, 36))
cmap = sns.diverging_palette(220, 10, as_cmap=True) 
# 绘制相关系数方阵热图
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap=cmap, ax=ax)
f.tight_layout()

def find_correlation(df, thresh=0.9): # df: 量化变数资料矩阵
    """
    Given a numeric pd.DataFrame, this will find highly correlated features,
    and return a list of features to be removed
    params:
    - df : pd.DataFrame
    - thresh : correlation coefficients threshold, will remove one of pairs of features with a correlation greater than this value
    - select_flat: a list of features to be removed
    """

    corrMatrix = df.corr()
    corrMatrix.loc[:,:] =  np.tril(corrMatrix, k=-1) # 取下三角矩阵

    already_in = set() # 集合结构避免重复计入相同元素
    result = []

    for col in corrMatrix:
        perfect_corr = corrMatrix[col][corrMatrix[col] > thresh].index.tolist() # Index物件转为list
        if perfect_corr and col not in already_in:
            already_in.update(set(perfect_corr)) # 更新集合
            perfect_corr.append(col)
            result.append(perfect_corr)

    select_nested = [f[1:] for f in result]
    select_flat = [i for j in select_nested for i in j]
    return select_flat


drop_list = find_correlation(cell_num, thresh=0.75) # 58 - 32 = 26
drop_list
len(drop_list) # 32

cell_num_filtered = cell_num.drop(drop_list, axis=1) # 后续建模可考虑此**原汁原味变数挑选**降维后的数据矩阵



