'''
Collated by Prof. Ching-Shih Tsou 邹庆士 教授 (Ph.D.) at the IDS, CICD, and CISD of NTUB (台北商业大学资讯与决策科学研究所暨智能控制与决策研究室教授兼校务永续发展中心主任); at the ME Dept. and CAIDS of MCUT (2020年借调至明志科技大学机械工程系担任特聘教授兼人工智慧暨资料科学研究中心主任两年); the CSQ (2019年起任品质学会大数据品质应用委员会主任委员); the DSBA (2013年创立台湾资料科学与商业应用协会); and the CARS (2012年创立中华R软体学会)
Notes: This code is provided without warranty.
'''

#### 2.1.2 Python语言资料排序
## ------------------------------------------------------------------------
import pandas as pd
# 印出pandas 套件版次
print(pd.__version__)
USArrests = pd.read_csv("./_data/USArrests.csv")
# 检视读档结果，注意奇怪栏名(Unnamed: 0)！
print(USArrests.head())

## ------------------------------------------------------------------------
# 修正栏位名称
# USArrests.rename(columns={'Unnamed: 0':'state'}, inplace=True) # 条条大路通罗马！
USArrests.columns = ['state', 'Murder', 'Assault',
'UrbanPop', 'Rape'] # 后四栏的栏位名重抄一遍，较不聪明！
# 设定state 为索引(index 从上面流水号变成下面州名)
USArrests = USArrests.set_index('state')
# Python 检视资料表前五笔数据，类似R 语言head(USArrests)
print(USArrests.head())
# Python 检视资料表的维度与维数(shape)
print(USArrests.shape)

## ------------------------------------------------------------------------
# 预设是依横向第一轴(axis = 0) 的索引名称升幂(ascending) 排列
print(USArrests.sort_index().head())
# 可调整为依纵向第二轴(axis = 1) 的索引名称降幂(descending) 排列
print(USArrests.sort_index(axis=1, ascending = False).head())
# 依Rape 栏位值，沿第一轴(axis = 0) 降幂排列
print(USArrests.sort_values(by="Rape", ascending=False).
head())
# 也可以依两栏位排序，前面栏位值平手时用后面栏位值排序
print(USArrests.sort_values(by=["Rape","UrbanPop"],
ascending=False).head())

# sort_values()平手状况理解
# USArrests.sort_values(by=["UrbanPop","Rape"], ascending=False) # 特别留意Arizona, Florida, Texas与Utah的UrbanPop名次(如何tie-breaking)

USArrests.loc[:,['UrbanPop',"Rape"]].sort_values(by=["UrbanPop","Rape"], ascending=False)


## ------------------------------------------------------------------------
# 沿第二轴(axis = 1) 同一观测值的四项事实数据名次
print(USArrests.rank(axis=1, ascending=False).head())
# 各栏位沿第一轴(axis = 0) 的五十州排名值
print(USArrests.rank(axis=0, ascending=False).head())

# rank()平手状况理解(method='average')
# USArrests.rank(axis=0, ascending=False) # 特别留意Arizona, Florida, Texas与Utah的UrbanPop名次均为10.5((9, 10, 11, 12) -> 10.5, (9+10+11+12)/4=10.5)
# tmp1 = USArrests.loc[:,['UrbanPop',"Rape"]].rank(axis=0, ascending=False)

# 同名时取最大名次值(method 预设为average)
print(USArrests.rank(axis=0, ascending=False,
method="max")[:10])

# rank()平手状况理解(method='max')
# tmp2 = USArrests.rank(axis=0, ascending=False, method="max") # (9, 10, 11, 12) -> 12

#### 2.1.4 Python语言资料变形
## ------------------------------------------------------------------------
USArrests = pd.read_csv("./_data/USArrests.csv")
# 变数名称调整
USArrests.columns = ['state', 'Murder', 'Assault',
'UrbanPop', 'Rape']
# pandas 宽表转长表(Python 语法中句点有特殊意义，故改为底线'_')
USArrests_dfl = (pd.melt(USArrests, id_vars=['state'],
var_name='fact', value_name='figure'))
print(USArrests_dfl.head())
# pandas 长表转宽表
# index 为横向变数，columns 为纵向变数，value 为交叉值
print(USArrests_dfl.pivot(index='state', columns='fact',
values='figure').head())

#### 2.1.6 Python语言资料清理
## ------------------------------------------------------------------------
algae = pd.read_csv("./_data/algae.csv")
# 单变量遗缺值检查
# R 语言语法可想成是head(isnull(algae['mxPH'])))
print(algae['mxPH'].isnull().head())
# 注意Python 输出格式化语法({} 搭配format() 方法)
print(" 遗缺{}笔观测值".format(algae['mxPH'].isnull().sum()))

## ------------------------------------------------------------------------
# 利用pandas 序列方法dropna() 移除单变量遗缺值
mxPH_naomit = algae['mxPH'].dropna()
print(len(mxPH_naomit))
# 检视整个资料表的遗缺状况
print(algae.isnull().iloc[45:55,:5])
# 横向移除不完整的观测值(200 笔移除16 笔)
algae_naomit = algae.dropna(axis=0)
print(algae_naomit.shape)

## ------------------------------------------------------------------------
# 以thresh 引数设定最低变数个数门槛(200 笔移除9 笔)
algae_over17 = algae.dropna(thresh=17)
print(algae_over17.shape)

## ------------------------------------------------------------------------
# 各变数遗缺状况：Chla 遗缺观测值数量最多, Cl 次之...
algae_nac = algae.isnull().sum(axis=0)
print(algae_nac)
# 各观测值遗缺状况：遗缺变数个数
algae_nar = algae.isnull().sum(axis=1)
print(algae_nar[60:65])

## ------------------------------------------------------------------------
# 检视不完整的观测值(algae_nar>0 回传横向遗缺数量大于0 的样本)
print(algae[algae_nar > 0][['mxPH', 'mnO2', 'Cl', 'NO3',
'NH4', 'oPO4', 'PO4', 'Chla']])
# 遗缺变数个数大于0(i.e. 不完整) 的观测值编号
print(algae[algae_nar > 0].index)
# 不完整的观测值笔数
print(len(algae[algae_nar > 0].index))
# 检视遗缺变数超过变数个数algae.shape[1] 之20% 的观测值
print(algae[algae_nar > algae.shape[1]*.2][['mxPH', 'mnO2',
'Cl', 'NO3', 'NH4', 'oPO4', 'PO4', 'Chla']])
# 如何获取上表的横向索引值？
print(algae[algae_nar > algae.shape[1]*.2].index)

## ------------------------------------------------------------------------
# 以drop() 方法，给IndexRange，横向移除遗缺严重的观测值
algae=algae.drop(algae[algae_nar > algae.shape[1]*.2].index)
print(algae.shape)

#### 2.2.3 Python语言群组与摘要
## ------------------------------------------------------------------------
# 载入必要套件
import pandas as pd
import numpy as np
import dateutil
# 载入csv 档
path = '.'
fname = '/_data/phone_data.csv'
data = pd.read_csv(''.join([path, fname])) # index_col = [0]
# 830 笔观测值，7 个变数
print(data.shape)
# 除index 与duration 外，所有栏位都是字串型别的类别变数
print(data.dtypes)
# 从编号1 的第2 栏位向后选，去除index 栏位
data = data.iloc[:,1:]
print(data.head())

## ------------------------------------------------------------------------
# 将日期字串逐一转为时间格式(pandas Series物件的apply()隐式回圈方法)
data['date'] = data['date'].apply(dateutil.parser.parse,
dayfirst=True)
# 也可以运用pandas 的to_datetime() 方法
data['date'] = pd.to_datetime(data['date']) # dayfirst=False, 请看说明文件的Warnings
# 'date' 的资料型别已改变
print(data.dtypes)

## ------------------------------------------------------------------------
# 传入原生串列物件创建pandas 序列，index 引数给定横向索引
series = pd.Series([20, 21, 12], index=['London',
'New York','Helsinki'])
print(series)
# pandas 序列物件apply() 方法的多种用法
# 可套用内建函数，例如：对数函数np.log()
print(series.apply(np.log))

# 也可以套用关键字为lambda 的匿名函数
# 其x 代表序列物件的各个元素
print(series.apply(lambda x: x**2))

# 或是自定义函数square()
def square(x):
    return x**2

print(series.apply(square))

# 另一个自定义函数，请注意参数custom_value 如何传入
def subtract_custom_value(x, custom_value):
    return x - custom_value

# 以args 引数传入值组(5,) 作为custom_value 参数
print(series.apply(subtract_custom_value, args=(5,)))

## ------------------------------------------------------------------------
# 检视变数名称(或是data.keys())
print(data.columns)
## Index(['index', 'date', 'duration', 'item', 'month',
## 'network', 'network_type'], dtype='object')

## ------------------------------------------------------------------------
# 服务类型次数分布
print(data['item'].value_counts())
# 网路服务型式次数分布
print(data['network_type'].value_counts())

## ------------------------------------------------------------------------
# 语音/数据最长服务时间
print(data['duration'].max())
# 语音通话的总时间计算，逻辑值索引+ 加总方法sum()
print(data['duration'][data['item'] == 'call'].sum())
# 每月记录笔数
print(data['month'].value_counts())

## ------------------------------------------------------------------------
# 网路营运商家数
print(data['network'].nunique())
# 网路营运商次数分布表
print(data['network'].value_counts())

## ------------------------------------------------------------------------
# 各栏位遗缺值统计
print(data.isnull().sum())

## ------------------------------------------------------------------------
# 依月份分组，先转为串列后仅显示最后一个月的分组数据
print(list(data.groupby(['month']))[-1])
# 分组数据是pandas 资料框的groupby 类型物件
print(type(data.groupby(['month'])))
# groupby 类型物件的groups 属性是字典结构
print(type(data.groupby(['month']).groups))
# 以年-月为各组数据的键，观测值索引为值
print(data.groupby(['month']).groups.keys())
## dict_keys(['2014-11', '2014-12', '2015-01', '2015-02',
## '2015-03'])
# '2015-03' 该组数据长度
print(len(data.groupby(['month']).groups['2015-03']))
# 取出'2015-03' 该组101 笔数据的观测值索引
print(data.groupby(['month']).groups['2015-03'])
# first() 方法取出各月第一笔资料，可发现各组数据栏位与原数据相同
print(data.groupby('month').first())

## ------------------------------------------------------------------------
# 各月电信服务总时数(Christmas 前很忙！)
print(data.groupby('month')['duration'].sum())

## ------------------------------------------------------------------------
# 各电信营运商语音通话总和
print(data[data['item'] == 'call'].groupby('network')
['duration'].sum())

## ------------------------------------------------------------------------
# 多个栏位分组，各月各服务类型的资料笔数统计
# 抓出分组数据的任何栏位统计笔数均可，此处以date 为例
print(data.groupby(['month', 'item'])['date'].count())

## ------------------------------------------------------------------------
# 分组统计结果pandas 序列(duration 变数名称在最下面)
print(data.groupby('month')['duration'].sum())
print(type(data.groupby('month')['duration'].sum()))
# 分组统计结果pandas 资料框(duration 变数名称在上方)
print(data.groupby('month')[['duration']].sum())
print(type(data.groupby('month')[['duration']].sum()))

## ------------------------------------------------------------------------
# 群组数据后的agg() 分组统计
print(data.groupby('month').agg({"duration": "sum"}))
# 分组索引值为分组变数month 的值
print(data.groupby('month').agg({"duration": "sum"}).index)
## Index(['2014-11', '2014-12', '2015-01', '2015-02',
## '2015-03'], dtype='object', name='month')
# 分组统计的栏位名称为duration
print(data.groupby('month').agg({"duration": "sum"}).columns)
# as_index=False 改变预设设定，month 从索引变成变数
print(data.groupby('month', as_index=False).agg({"duration":
"sum"}))

## ------------------------------------------------------------------------
# 各组多个统计计算
# 各月(month) 各服务(item) 的服务时间、网路服务型式与日期统计
print(data.groupby(['month', 'item']).agg({'duration':
[min, max, sum], 'network_type': "count", 'date':
[min, 'first', 'nunique']}))
#                 duration                        network_type
#                      min        max         sum        count
# month     item
# 2014-11   call     1.000   1940.000   25547.000          107
#           data    34.429     34.429     998.441           29
#           sms      1.000      1.000      94.000           94
# 2014-12   call     2.000   2120.000   13561.000           79
#           data    34.429     34.429    1032.870           30
#           sms      1.000      1.000      48.000           48
# 2015-01   call     2.000   1859.000   17070.000           88
#           data    34.429     34.429    1067.299           31
#           sms      1.000      1.000      86.000           86
# 2015-02   call     1.000   1863.000   14416.000           67
#           data    34.429     34.429    1067.299           31
#           sms      1.000      1.000      39.000           39
# 2015-03   call     2.000  10528.000   21727.000           47
#           data    34.429     34.429     998.441           29
#           sms      1.000      1.000      25.000           25
#
#
#                               date
# month     item                 min   first           nunique
# 2014-11   call 2014-10-15 06:58:00   2014-10-15 06:58:00 104
#           data 2014-10-15 06:58:00   2014-10-15 06:58:00 29
#           sms  2014-10-16 22:18:00   2014-10-16 22:18:00 79
# 2014-12   call 2014-11-14 17:24:00   2014-11-14 17:24:00 76
#           data 2014-11-13 06:58:00   2014-11-13 06:58:00 30
#           sms  2014-11-14 17:28:00   2014-11-14 17:28:00 41
# 2015-01   call 2014-12-15 20:03:00   2014-12-15 20:03:00 84
#           data 2014-12-13 06:58:00   2014-12-13 06:58:00 31
#           sms  2014-12-15 19:56:00   2014-12-15 19:56:00 58
# 2015-02   call 2015-01-15 10:36:00   2015-01-15 10:36:00 67
#           data 2015-01-13 06:58:00   2015-01-13 06:58:00 31
#           sms  2015-01-15 12:23:00   2015-01-15 12:23:00 27
# 2015-03   call 2015-02-12 20:15:00   2015-02-12 20:15:00 47
#           data 2015-02-13 06:58:00   2015-02-13 06:58:00 29
#           sms  2015-02-19 18:46:00   2015-02-19 18:46:00 17

# 请分析不同行动通讯服务下，服务时间(duration)的平均值、中位数及标准差，并说明可能的发现。
data.groupby(['item']).agg({'duration':[np.mean, np.median, np.std]})


####        2.3.2.1 奇异值矩阵分解
## ------------------------------------------------------------------------
# 载入阵列与奇异值分解类别
from numpy import array
from scipy.linalg import svd
A = array([[1, 2], [3, 4], [5, 6]])
print(A)
# 奇异值矩阵分解，输出U, s, VT 三个方阵或矩阵
U, s, VT = svd(A)
print(U)
# 稍后以s 中的两个值产生3*2 对角矩阵
print(s)
print(VT)

## ------------------------------------------------------------------------
# numpy 套件与矩阵代数密切相关
from numpy import diag
# 点积运算方法
from numpy import dot
# 零值矩阵创建
from numpy import zeros
# 创建m*n 阶Sigma 矩阵，预存值为零
Sigma = zeros((A.shape[0], A.shape[1]))
# 对Sigma 矩阵植入2*2 对角方阵
Sigma[:A.shape[1], :A.shape[1]] = diag(s)
print(Sigma)
# 点积运算重构原矩阵
B = U.dot(Sigma.dot(VT))
print(B)

## ------------------------------------------------------------------------
# 3*3 方阵
A = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(A)
# SVD 方阵分解
U, s, VT = svd(A)
print(U)
# 中间的Sigma 亦为方阵
Sigma = diag(s)
print(Sigma)
print(VT)
# 点积运算重构原矩阵
B = U.dot(Sigma.dot(VT))
print(B)

## ------------------------------------------------------------------------
# 3*10 矩阵
A = array([
[1,2,3,4,5,6,7,8,9,10],
[11,12,13,14,15,16,17,18,19,20],
[21,22,23,24,25,26,27,28,29,30]])
# SVD 分解
U, s, VT = svd(A)
# 创建m*n 阶矩阵，预存值为零
Sigma = zeros((A.shape[0], A.shape[1]))
# 对Sigma 矩阵植入对角方阵
Sigma[:A.shape[0], :A.shape[0]] = diag(s)
# 以前两个最大的奇异值做SVD 近似计算
n_elements = 2
Sigma = Sigma[:, :n_elements]
VT = VT[:n_elements, :]
# 计算近似矩阵B((3*3).(3*2).(2*10))
B = U.dot(Sigma.dot(VT))
print(B)

## ------------------------------------------------------------------------
# SVD 降维运算((3*3) * (3*2))
T = U.dot(Sigma)
print(T)
# 另一种SVD 降维运算方式((3*10).(10*2))
T = A.dot(VT.T)
print(T)

## ------------------------------------------------------------------------
# 载入sklearn 的SVD 分解降维运算类别
from sklearn.decomposition import TruncatedSVD
# 3*10 矩阵
A = array([
[1,2,3,4,5,6,7,8,9,10],
[11,12,13,14,15,16,17,18,19,20],
[21,22,23,24,25,26,27,28,29,30]])
# 宣告SVD 分解降维空模
svd = TruncatedSVD(n_components=2)
# 配适实模
svd.fit(A)
# 转换应用
result = svd.transform(A)
print(result)

