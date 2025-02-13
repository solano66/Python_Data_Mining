'''
Collated by Prof. Ching-Shih Tsou 邹庆士 教授 (Ph.D.) at the IDS, CICD, and CISD of NTUB (台北商业大学资讯与决策科学研究所暨智能控制与决策研究室教授兼校务永续发展中心主任); at the ME Dept. and CAIDS of MCUT (2020年借调至明志科技大学机械工程系担任特聘教授兼人工智慧暨资料科学研究中心主任两年); the CSQ (2019年起任品质学会大数据品质应用委员会主任委员); the DSBA (2013年创立台湾资料科学与商业应用协会); and the CARS (2012年创立中华R软体学会)
Notes: This code is provided without warranty.
'''

#### 4.1 资料视觉化
## ------------------------------------------------------------------------
# 载入必要套件，并记为简要的名称
import matplotlib.pyplot as plt
import numpy as np
# 产生或汇入资料
x = np.arange(0, 10, 0.2)
y = np.sin(x)
# 产生图形(pyplot 语法)
plt.plot(x, y)
# 将图形显示在荧幕上
# plt.show()

# 载入必要套件，并记为简要的名称
import matplotlib.pyplot as plt
import numpy as np
# 产生或汇入资料
x = np.arange(0, 10, 0.2)
y = np.sin(x)
# 产生图形(物件导向语法)
fig = plt.figure()
ax = fig.add_subplot(1,1,1) # (列，行，图)
ax.plot(x, y)
# 将图形显示在荧幕上
# plt.show()
# 图形储存方法savefig()
# fig.savefig('./_img/plt.png', bbox_inches='tight')

# 载入必要套件pylab(过时！请跳过)
from pylab import *
# 产生或汇入资料
x = np.arange(0, 10, 0.2)
y = np.sin(x)
# 产生图形(Matlab 语法)
plot(x, y)
# 将图形显示在荧幕上
# show()

# 载入必要套件，并记为简要的名称
import matplotlib.pyplot as plt
import numpy as np
# 产生或汇入资料
x = np.arange(0, 10, 0.2)
y = np.sin(x)
z = np.cos(x)
# 产生图面与子图(由此一次执行到plt.show()之前)
fig, axs = plt.subplots(nrows=2, ncols=1)
# 绘制第一个子图正弦波，加上垂直轴标签
axs[0].plot(x, y) # 高阶绘图
axs[0].set_ylabel('Sine') # 低阶绘图
# 绘制第二个子图余弦波，加上垂直轴说明文字
axs[1].plot(x, z) # 高阶绘图
axs[1].set_ylabel('Cosine') # 低阶绘图
# 将图形显示在荧幕上
# plt.show()
# 图形储存方法savefig()
# fig.savefig('./_img/multiplt.png', bbox_inches='tight')
# 还原图形与子图的预设设定
fig, ax = plt.subplots(nrows=1, ncols=1)

# 载入Python 语言pandas 套件与生化资料集
import pandas as pd
path = './_data/'
fname = 'segmentationOriginal.csv'
# 中间无任何空白的方式连结路径与档名
cell = pd.read_csv("".join([path, fname]))

# 119 个变数
print(len(cell.columns))

# 挑选五个量化变数
partialCell = cell[['AngleCh1', 'AreaCh1', 'AvgIntenCh1',
'AvgIntenCh2', 'AvgIntenCh3']]
# 以pandas 资料框的boxplot() (简便绘图)方法绘制并排盒须图(图4.3)
ax = partialCell.boxplot() # partialCell 是资料框物件
# pandas 图形须以get_figure() 方法取出图形后方能储存
fig = ax.get_figure()
# fig.savefig('./_img/pd_boxplot.png')

# 以seaborn 套件的boxplot() 函数绘制并排盒须图
import seaborn as sns
ax = sns.boxplot(x="variable", y="value",
data=pd.melt(partialCell)) # 10095 rows = 2019 samples * 5 variables

# 宽表转长表自动生成的变数名称variable 与value
print(pd.melt(partialCell)[2015:2022])

# seaborn 图形也须以get_figure() 方法取出图形后方能储存
fig = ax.get_figure()
# fig.savefig('./_img/sns_boxplot.png')

# 载入Python 图形文法绘图套件及其内建资料集(建议用Python虚拟环境，搭配matplotlib 3.1.3版本，以及pandas 0.22.0版本，方能安装ggplot与plotnine)
from ggplot import * # A messy environment gotten after such importing (!conda install -c conda-forge ggplot --y)

# 检视钻石资料集前5 笔样本
import pandas as pd
print(diamonds.iloc[:, :9].head())

# 数值与类别变数混成的资料集
print(diamonds.dtypes)

# 图形文法的图层式绘图
p = ggplot(aes(x='price', color='clarity'), data=diamonds) + geom_density() + scale_color_brewer(type='div') + facet_wrap('cut')
p

# ggplot 储存图形方法save()
p.save('./_img/gg_density.png')

#### ggplot绘图部份建议执行下方程式码，以使环境较不混乱！
# 载入Python 图形文法绘图套件及其内建资料集
# import ggplot as gp
import plotline as gp # !conda install -c conda-forge plotnine --y
from plotnine.data import diamonds

# 检视钻石资料集前5 笔样本
import pandas as pd
# print(gp.diamonds.iloc[:, :9].head())
print(diamonds.iloc[:, :9].head())

# 数值与类别变数混成的资料集
# print(gp.diamonds.dtypes)
print(diamonds.dtypes)

# 图形文法的图层式绘图(底部图层+密度曲线图层+色盘控制图层+分面绘制图层)
# p = gp.ggplot(gp.aes(x='price', color='clarity'), data=gp.diamonds) + gp.geom_density() + gp.scale_color_brewer(type='div') + gp.facet_wrap('cut')
p = gp.ggplot(data=gp.diamonds, gp.aes(x='price', color='clarity')) + gp.geom_density() + gp.scale_color_brewer(type='div') + gp.facet_wrap('cut')
p

# ggplot 储存图形方法save()
p.save('./_img/gg_density.png')
###

#### 4.2.2 线上音乐城关联规则分析
## ------------------------------------------------------------------------
import pandas as pd
# 设定pandas 横列与纵行结果呈现最大宽高值
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
# 线上音乐城聆听记录载入
lastfm = pd.read_csv("./_data/lastfm.csv")
# 聆听历程长资料
print(lastfm.head())

# 检视栏位资料型别，大多是类别变数
print(lastfm.dtypes)

# 统计各用户线上聆听次数(一维频次表)
print(lastfm.user.value_counts()[:5])

# 独一无二的用户编号长度，共有15000 位用户
print(lastfm.user.unique().shape) # lastfm.user.nunique()

# 各艺人被点阅次数
print(lastfm.artist.value_counts()[:5])

# 确认演唱艺人人数，共有1004 位艺人
print(lastfm.artist.unique().shape) # lastfm.artist.nunique()

# 依用户编号分组，grouped内有15,000个子表
grouped = lastfm.groupby('user')

# 检视前两组的子表，前两位用户各聆听16 与29 位艺人专辑
print(list(grouped)[:2])

# 用户编号有跳号现象
print(list(grouped.groups.keys())[:10])

# 以agg() 方法传入字典，统计各使用者聆听艺人数
numArt = grouped.agg({'artist': "count"})
print(numArt[5:10])

# 取出分组表艺人名称一栏
grouped = grouped['artist']
# Python 串列推导，拆解分组资料为(巢状nested或嵌套embedded)串列(长表 -> 宽表)
music = [list(artist) for (user, artist) in grouped]

# 限于页面宽度，取出交易记录长度<3 的数据呈现巢状串列的整理结果
print([x for x in music if len(x) < 3][:2])

from mlxtend.preprocessing import TransactionEncoder # !conda install -c conda-forge mlxtend --y
# pip install mlxtend --proxy="http://yourproxy:portnumber"
# 交易资料格式编码(同样是宣告空模 -> 配适实模-> 转换运用)
te = TransactionEncoder()
# 传回numpy 二元值矩阵txn_binary
txn_binary = te.fit(music).transform(music)
# 检视交易记录笔数与品项数
print(txn_binary.shape)

# 读者自行执行dir()，可以发现te 实模物件下有columns_ 属性
# dir(te)
# 检视部分品项名称
print(te.columns_[15:20])

# numpy 矩阵组织为二元值资料框(非常稀疏！False远多于True)
df = pd.DataFrame(txn_binary, columns=te.columns_)
print(df.iloc[:5, 15:20])

# apriori 频繁品项集(或强物项集)探勘(演算法)
from mlxtend.frequent_patterns import apriori
# pip install --trusted-host pypi.org mlxtend

# 挖掘时间长，因此记录执行时间
# 可思考为何R 语言套件{arules} 的apriori() 快速许多？
import time
start = time.time()
freq_itemsets = apriori(df, min_support=0.01,
use_colnames=True)
end = time.time()
print(end - start)

# apply() 结合匿名函数统计品项集长度，并新增'length' 栏位于后
freq_itemsets['length'] = freq_itemsets['itemsets'].apply(lambda x: len(x))
# 频繁品项集资料框，支持度、品项集与长度
print(freq_itemsets.head())

print(freq_itemsets.dtypes)

# 布林值索引筛选频繁品项集
print(freq_itemsets[(freq_itemsets['length'] == 2)
& (freq_itemsets['support'] >= 0.05)])

# association_rules 关联规则集生成
from mlxtend.frequent_patterns import association_rules
# 从频繁品项集中产生49 条规则(生成规则confidence >= 0.5)
musicrules = association_rules(freq_itemsets,
metric="confidence", min_threshold=0.5)
print(musicrules.head())

# apply() 结合匿名函数统计各规则前提部长度
# 并新增'antecedent_len' 栏位于后
musicrules['antecedent_len'] = musicrules['antecedents'].apply(lambda x: len(x))
print(musicrules.head())

# 布林值索引筛选关联规则
print(musicrules[(musicrules['antecedent_len'] > 0) &
(musicrules['confidence'] > 0.55)&(musicrules['lift'] > 5)])

#### 铁达尼号资料集练习(补充)
import pandas as pd
titanic = pd.read_csv('./_data/Titanic.csv')
titanic = titanic.drop(['Unnamed: 0'], axis=1)
tf = pd.get_dummies(titanic) # 预设是单热编码

from mlxtend.frequent_patterns import apriori

freq_itemsets = apriori(tf, min_support=0.1, use_colnames=True)

from mlxtend.frequent_patterns import association_rules
# 从频繁品项集中产生关联规则(生成规则confidence >= 0.5)
titanicrules = association_rules(freq_itemsets,
metric="confidence", min_threshold=0.5)
# 关心结果部是frozenset({'Survived_Yes'})以及frozenset({'Survived_No'})的规则

#### 4.3.1 k平均数集群
## ------------------------------------------------------------------------
# library(animation) # Please try it in R!
# kmeans.ani() # Please try it in R!

#### 4.3.1.1 青少年市场区隔案例
## ------------------------------------------------------------------------
import numpy as np
import pandas as pd
teens = pd.read_csv("./_data/snsdata.csv")
# 文件词项矩阵前面加上入学年、性别、年龄与朋友数等栏位
print(teens.shape) # 30000 * (4 + 36 terms by NLP)

# 留意gradyear 的资料型别
print(teens.dtypes)

# gradyear 更新为字串str 型别
teens['gradyear'] = teens['gradyear'].astype('str')
# 除了资料型别外，ftypes 还报导了属性向量是稀疏还是稠密的
# print(teens.ftypes.head()) # deprecated after 1.0.0

# 各变数叙述统计值(报表过宽，只呈现部份结果)
print(teens.describe(include='all'))

# 各栏位遗缺值统计(只有gender 与age 有遗缺)(注意！此处并未进行遗缺值处理)
print(teens.isnull().sum().head())

# 各词频变数标准化建模
from sklearn.preprocessing import StandardScaler # Step 1

sc = StandardScaler() # Object-oriented programming paradigm, # Step 2

# 配适与转换接续完成函数
teens_z = sc.fit_transform(teens.iloc[:,4:]) # Steps 3 & 4

# 错误用法！Python的类别函数不可使用泛函式编程语法
# teens_z = StandardScaler(teens.iloc[:,4:]) # > scikit-learn 0.23.2 可以！

# scikit-learn下preprocessing模组的scale()函数可用泛函式编程语法
from sklearn.preprocessing import scale
teens_z = scale(teens.iloc[:,4:])

# 资料导向程式设计经常输出与输入不同调(DataFrame 入ndarray 出)
print(type(teens_z))

# 转为资料框物件取用describe() 方法确认标准化结果
print(pd.DataFrame(teens_z[:,30:33]).describe())
# mean 5.494864e-17  1.136868e-17 -9.687066e-17 可能数字上会有差异，这说明二进位制的计算机的数值运算不稳定性(numerical instability)

# Python k 平均数集群，随机初始化的集群结果通常比较好
from sklearn.cluster import KMeans # Step 1
mdl = KMeans(n_clusters=5, init='random') # Step 2

# 配适前空模的属性与方法
pre = dir(mdl)
# 空模的几个属性与方法
print(pre[51:56])
# 以标准化文件词项矩阵配适集群模型
import time
start = time.time()
mdl.fit(teens_z) # Step 3
end = time.time()
print("k-Means fitting spent {} seconds".format(end - start))

# 配适后实模的属性与方法
post = dir(mdl)
# 实模的几个属性与方法
print(post[51:56])

# 实模与空模属性和方法的差异(前或后有下底线_)
print(list(set(post) - set(pre)))

# sklearn 模型的存出(dump)与读入(load)
import pickle
filename = './_data/kmeans.sav' # 设定存出路径与档名
pickle.dump(mdl, open(filename, 'wb')) # 'wb': write out in binary
res = pickle.load(open(filename, 'rb')) # 'rb': read in in binary
res = mdl
# res.labels_ 为30,000 名训练样本的归群标签
# import sys
# np.set_printoptions(threshold=sys.maxsize)
print(res.labels_.shape)

# 五群人数分布(思考numpy下如何做！)
print(pd.Series(res.labels_).value_counts())

# 前10 个样本的群编号
print (res.labels_[:10])

# 各群字词平均词频矩阵的维度与维数
print(res.cluster_centers_.shape)

# 转换成pandas 资料框，给予群编号与字词名称，方便结果诠释
cen = pd.DataFrame(res.cluster_centers_, index = range(5),
columns = teens.iloc[:,4:].columns)
print(cen)

# 每次归群结果的释义会有不同
# Princesses: 3
# Criminals: 0
# Basket Cases: 4
# Athletes: 2
# Brains: 1

# 各群中心座标矩阵转置后绘图
ax = cen.T.plot() # seaborn, ggplot or pandas ?
# 低阶绘图设定x 轴刻度位置
ax.set_xticks(list(range(36)))
# 低阶绘图设定x 轴刻度说明文字
ax.set_xticklabels(list(cen.T.index), rotation=90)
fig = ax.get_figure()
fig.tight_layout()
# fig.savefig('./_img/sns_lineplot.png')

# 以下为课本/讲义没有的补充程式码，主要在进行事后(建模后)的分析
# 添加群编号于原资料表后
teens = pd.concat([teens, pd.Series(mdl.labels_).rename('cluster')], axis=1) # axis=1 means cbind() in R

# 抓集群未使用的三个变量(刚才归群时未用，但事后分析确有助于了解各群的异同，以及归群结果的品质)
teens[['gender','age','friends','cluster']][0:5]

# 各群平均年龄(群组与摘要也！)
teens.groupby('cluster').aggregate({'age': np.mean}) # 同侪间年龄差异不大！

# 新增是否为女生栏位'female'
teens.gender.value_counts()
teens.gender.value_counts(dropna = False)

# Equivalent of R/ifelse in Python/Pandas? Compare string columns? (https://stackoverflow.com/questions/35666272/equivalent-of-r-ifelse-in-python-pandas-compare-string-columns#)
def if_this_else_that(x, list_of_checks, yes_label, no_label):
    if x in list_of_checks:
        res = yes_label
    else: 
        res = no_label
    return(res)

teens['female'] = teens['gender'].apply(lambda x: if_this_else_that(x, ['F'], True, False))

teens[['gender', 'female']].head(n=20)

teens.female.sum() # 22054

# 各群女生人数比例(群组与摘要也！)
teens.groupby('cluster').aggregate({'female': np.mean})

teens.gender.value_counts()

# 各群朋友数(群组与摘要也！)
teens.groupby('cluster').aggregate({'friends': np.mean})

#### 4.3.3.1 密度集群案例
# 载入Python 密度集群类别DBSCAN()
from sklearn.cluster import DBSCAN # Density-Based Spatial Clustering of Applications with Noise
import numpy as np
import pandas as pd
# 读取批发客户资料集
data = pd.read_csv("./_data/wholesale_customers_data.csv")

# 注意各变数实际意义，而非只看表面上的数字
print(data.head())

print(data.dtypes)

# 移除名目尺度类别变数
data.drop(["Channel", "Region"], axis = 1, inplace = True)

# 二维空间可视觉化集群结果
data = data[["Grocery", "Milk"]]
# 集群前资料须标准化
data = data.values.astype("float32", copy = False)
from sklearn.preprocessing import StandardScaler
stscaler = StandardScaler().fit(data)
data = stscaler.transform(data)
# 密度集群前，先绘制标准化后的样本散布图
ax = pd.DataFrame(data, columns=["Grocery", "Milk"]).plot.scatter("Grocery", "Milk")
fig = ax.get_figure()
# fig.savefig('./_img/normalized_scatter.png')

# 以标准化资料配适DBSCAN 集群模型
dbsc = DBSCAN(eps = .5, min_samples = 15).fit(data)
# 归群结果存出
labels = dbsc.labels_
# 杂讯样本的群标签为-1(numpy 产制次数分布表的方式)
print(np.unique(labels, return_counts=True))

# # 设定绘图颜色值阵列(书本上原绘图程式码)
# colors = np.array(['purple', 'blue'])
# # 利用labels+1 给定各样本描点颜色
# ax = pd.DataFrame(data, columns=["Grocery", "Milk"]).plot.scatter("Grocery", "Milk", c=colors[labels+1])
# fig = ax.get_figure()
# # fig.savefig('./_img/dbscan_scatter.png')

#### 以下是绘图加强版补充程式码
# 密度集群结果资料框
import string
df = pd.DataFrame(data, columns=["Grocery", "Milk"])
df["Cluster"] = pd.Series(labels).apply(lambda x: string.ascii_lowercase[x+1])

# 设定绘图点形、颜色与说明文字字典
marker_dict = {'a':'^', 'b':'o'} # '^': 三角点形; 'o': 圆形点形
color_dict = {'a':'blue', 'b':'red'}
label = {'a':'noise', 'b':'dense'}

# 依集群结果群组资料
groups = df.groupby(['Cluster']) # -1与0两群

# 分组绘制不同点形与颜色的散布图
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.margins(0.05)
for name, group in groups:
    marker = marker_dict[name]
    color = color_dict[name]
    ax.plot(group.Grocery, group.Milk, marker=marker, linestyle='', label=label[name], color=color) # ms=12, 
ax.legend()
ax.set_xlabel('Grocery')
ax.set_ylabel('Milk')
plt.show()
# fig.savefig('./_img/dbscan_scatter.png')
