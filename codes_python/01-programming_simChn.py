'''
Collated by Prof. Ching-Shih Tsou 邹庆士 教授 (Ph.D.) at the IDS, CICD, and CISD of NTUB (台北商业大学资讯与决策科学研究所暨智能控制与决策研究室教授兼校务永续发展中心主任); at the ME Dept. and CAIDS of MCUT (2020年借调至明志科技大学机械工程系担任特聘教授兼人工智慧暨资料科学研究中心主任两年); the CSQ (2019年起任品质学会大数据品质应用委员会主任委员); the DSBA (2013年创立台湾资料科学与商业应用协会); and the CARS (2012年创立中华R软体学会)
Notes: This code is provided without warranty.
'''

#### 1.4.1 Python语言原生资料物件操弄(新版Spyder请再增加一个#)
## ------------------------------------------------------------------------
# 中括弧创建Python 串列/串行/列表list，千万别与R 串列混为一谈!
x = [1,3,6,8]
print(x)

# type(x) # It's a list

# dir()是查询物件x下有何可用属性与方法的重要函数
dir(x)

## ------------------------------------------------------------------------
# 可以混型存放，参见图1.8 Python 索引编号从0 开始
x[1] = 'peekaboo'
print(x)
# Python 句点语法，引用串列物件append() 方法
# 添加传入的元素于串列末端
x.append('dwarf') # 一个传入值
print(x)
# insert() 方法在指定位置(前)塞入元素
x.insert(1, 'Safari') # 两个传入值
print(x)

# Python - dir() - how can I differentiate between functions/method and simple attributes? (https://stackoverflow.com/questions/26818007/python-dir-how-can-i-differentiate-between-functions-method-and-simple-att)
[(name,type(getattr(x,name))) for name in dir(x)]

# pop() 方法将指定位置上的元素移除
x.pop(2)
print(x)
# 以in 关键字判断，序列型别物件中是否包含某个元素
print('Safari' in x)
# 串列串接
print([4, 'A_A', '>_<'] + [7, 8, 2, 3])
# 排序
a = [7, 2, 5, 1, 3]
# 预设对数值升幂(预设)排序
print(sorted(a))
# 透过字串长度升幂(预设) 排序
b = ['saw', 'small', 'He', 'foxes', 'six']
# 串列物件b 为sorted() 函数的位置(positional) 参数值
# key 为sorted 函数的关键字(keyword) 参数,len 是关键字参数值
# Python 函数的位置参数必须在关键字参数前
# 参见1.6.2 节Python 语言物件导向
print(sorted(b, key=len))

## ------------------------------------------------------------------------
# 小括弧创建Python 值组
y = (1, 3, 5, 7)
print(y)
# 可以省略小括弧
y = 1, 3, 5, 7
print(y)
# 值组中还有值组，称为巢状值组(或称嵌套式值组，串列也可以！勿忘Python计算机科学家的语言，所以资料结构弹性较大)
nested_tup = (4, 5, 6), (7, 8)
print(nested_tup)
# 透过tuple 函数可将序列或迭代物件转为值组
tup = tuple(['foo', [1, 2], True])
print(tup)
# 值组是不可更改的(immutable)
# tup[2] = False
# TypeError: 'tuple' object does not support item assignment
# 但是值组tup 的第二个元素仍为可变的(mutable) 串列
tup[1].append(3) # This's the append method for list, not tuple !
print(tup)
# 解构(unpacking) 值组
tup = (4, 5, 6)
a, b, c = tup
print(c)
# Python 的变数交换方式
x, y = 1, 2
x, y = y, x
print(x)
print(y)

## ------------------------------------------------------------------------
# 大括弧创建字典
d1 = {'a' : 'some value', 'b' : [1, 2, 3, 4]}
print(d1)
# 字典新增元素方式
d1['c'] = 'baby'
d1['dummy'] = 'another value'
print(d1)
## {'a': 'some value', 'b': [1, 2, 3, 4], 'c': 'baby',
## 'dummy': 'another value'}
# 字典取值(没有编号index的概念)
print(d1['b'])
# 字典物件get() 方法可以取值，查无该键时回传'There does
# not have this key.'
print(d1.get('b', 'There does not have this key.'))
# 例外状况发生
print(d1.get('z', 'There does not have this key.'))
# 判断字典中是否有此键
print('b' in d1)
print('z' in d1)
# 字典物件pop() 方法可以删除元素，例外处理同get() 方法
print(d1.pop('b','There does not have this key.'))
# 键为'b' 的字典元素被移除了
print(d1)
# 例外状况发生
print(d1.pop('z','There does not have this key.'))
# 取得dict 中所有keys，常用！
print(d1.keys())
# 以list() 方法转为串列物件，注意与上方结果的差异，后不赘述！
print(list(d1.keys()))
# 取得dict 中所有values
print(d1.values())
print(list(d1.values()))
# 取得dict 中所有的元素(items)，各元素以tuple 包着key 及
# value
print(d1.items())
## dict_items([('a', 'some value'), ('c', 'baby'),
## ('dummy', 'another value')])
# 将两个dict 合并，后面更新前面
x = {'a':1,'b':2} # 註：课本中为a
y = {'b':0,'c':3} # 註：课本中为b
x.update(y) # 註：课本中为a.update(b)
print(x) # 註：课本中为print(a)
# 两个串列分别表示keys 与values
# 以拉链函数zip() 将对应元素捆绑后转换为dict
tmp = dict(zip(['name','age'], ['Tommy',20]))
print(tmp)

## ------------------------------------------------------------------------
# set() 函数创建Python 集合物件
print(set([2, 2, 2, 1, 3, 3]))
# 同前不计入重复的元素，所以还是1, 2, 3
print({1, 2, 3, 3, 3, 1})
# 集合物件联集运算union (or)
a = {1, 2, 3, 4, 5}
b = {3, 4, 5, 6, 7, 8}
print(a | b)
# 集合物件交集运算intersection (and)
print(a & b)
# 集合物件差集运算difference
print(a - b)
# print(b - a) # {8, 6, 7}，差集是非对称运算
# 集合物件对称差集(或逻辑互斥) 运算symmetric difference(xor)
print(a ^ b)
# 判断子集issubset() 方法
a_set = {1, 2, 3, 4, 5}
print({1, 2, 3}.issubset(a_set))
# 判断超集issuperset() 方法
print(a_set.issuperset({1, 2, 3}))
# 判断两集合是否相等== 运算子
print({1, 2, 3} == {3, 2, 1})
# 判断两值组是否不等!= 运算子
print({1, 2, 3} != {3, 2, 1})

#### 1.4.2 Python语言衍生资料物件取值
## ------------------------------------------------------------------------
# 载入numpy 套件并简记为np，方便后续引用
import numpy as np
# 呼叫arange() 方法(类似R 语言seq() 函数)，
# 并结合reshape() 方法创建ndarray 物件(4 横列5 纵行)
data = np.arange(20, dtype='int32').reshape((4, 5)) # Python预设是横列row导向，而R预设是纵行column导向
print(data)
# numpy ndarray 类别
print(type(data))

## ------------------------------------------------------------------------
# start:end[:step]
# 属性矩阵Ｘ与反应变数y切分
# 留意X 取至倒数第一纵行(前包后不包)，以及y 只取最后一行
X, y = data[:, :-1], data[:, -1] # 两次指派合而为一
print(X)
print(y)
# y_2D = y.reshape((-1,1))
## ------------------------------------------------------------------------
# 一维取单一元素
print(X[2]) # 取第三横列
# 二维取单一横列，结果同上
print(X[2,:])
# 一维取值从给定位置至最末端
# 中括弧取值时同R 语言一样运用冒号(start:end) 运算子
# 冒号(start:end) 后方留空代表取到尽头
print(X[2:])
# 二维取值，结果同上
print(X[2:,:])
# 倒数的负索引与间距(从倒数第三纵行取到最末行，取值间距为2)
print(X[2:,-3::2]) # start:end:step (X[2:,-3:4:2]或X[2:,-3:3:2])

## ------------------------------------------------------------------------
import pandas as pd
# 第一次使用pandas的read_excel方法需安装xlrd套件(!conda install xlrd --y)
# skiprows=1 表示从第2 横列开始读取资料(请自行更换读档路径)
fb=pd.read_excel('./_data/facebook_checkins_2013-08-24.xls', skiprows=1)

## ------------------------------------------------------------------------
# 确认其为pandas 资料框物件
type(fb) # pandas.core.frame.DataFrame

# type(fb.longitude) # pandas.core.series.Series

# type(fb.longitude.values) # numpy.ndarray

# dir(fb.longitude) # 有'to_list'方法

# type(fb.longitude.to_list()) # list

# help(fb.longitude.to_list)

# 查询物件fb 的属性与方法，内容过长返回部分结果
print(dir(fb)[-175:-170])

## ------------------------------------------------------------------------
# 以pandas DataFrame 物件的dtypes 属性检视各栏位资料型别
print(fb.dtypes)

## ------------------------------------------------------------------------
# 请与R 比较语法异同及结果差异
print(fb.tail(n=10)) # tail(fb, n = 10)预设看后五笔样本

#### A. 以下是以中括弧对DataFrame取值
## ------------------------------------------------------------------------
# 二维资料框取出一维序列，无栏位名称
print(fb['地标名称'].head())
# pandas 一维结构Series
print(type(fb['地标名称']))
# 双中括弧取出的物件仍为二维结构，有栏位名称
print(fb[['地标名称']].head())
# pandas 二维结构DataFrame
print(type(fb[['地标名称']]))

#### B. 以下是以句点语法对DataFrame取值
## ------------------------------------------------------------------------
# 资料框句点语法取值，无栏位名称
print(fb.类别.head())
# pandas 一维结构Series(句点语法无法取出二维单栏物件！)
print(type(fb.类别))

#### C. 以下是中括弧搭配loc及iloc方法对DataFrame取值
## ------------------------------------------------------------------------
# 资料框loc 方法取值(注意此处冒号运算子为前包后也包!Seeing is believing.)
# loc 方法用中括弧！！！
print(fb.loc[:10, ['地区','累积打卡数']])

## ------------------------------------------------------------------------
# 资料框iloc() 方法取值(注意此处冒号运算子却又是前包后不包! Seeing is believing.)
print(fb.iloc[:10, [6, 2]]) # iloc中i 之意是丢索引编号(index)而非字串

## ------------------------------------------------------------------------
# 过时用法在此未执行，因为超过过渡期后会产生错误讯息
print(fb.ix[:10, ['latitude', 'longitude']])
print(fb.ix[:10, [3, 4]])

#### 1.4.3 Python语言类别变数编码
## ------------------------------------------------------------------------
import pandas as pd
# 以原生资料结构巢状串列建构pandas 资料框
df = pd.DataFrame([['green', 'M', 10.1, 'class1'], ['red',
'L', 13.5, 'class2'], ['blue', 'XL', 15.3, 'class1']])
# 设定资料框栏位名称
df.columns = ['color', 'size', 'price', 'classlabel']
print(df)

## ------------------------------------------------------------------------
# 定义编码规则字典
size_mapping = {'XL': 3, 'L': 2, 'M': 1}
# 序列map() 方法完成编码，并更新size 变数
df['size'] = df['size'].map(size_mapping)
print(df)

## ------------------------------------------------------------------------
# 载入类别
from sklearn.preprocessing import LabelEncoder
# 创建(或称实作) 类别物件class_le
class_le = LabelEncoder()
# 传入类别变数进行配适与转换
y = class_le.fit_transform(df['classlabel'])
# 标签编码完成(对应整数值预设从0 开始)
print(y)
# y = LabelEncoder().fit_transform(df['classlabel'])
## ------------------------------------------------------------------------
# 逆转换回原类别值
print(class_le.inverse_transform(y.reshape(-1, 1)))
# 注意下面两个资料物件内涵相同，但维度不同！前一维，后二维
print(y)
print(y.reshape(-1, 1))

## ------------------------------------------------------------------------
# 取出欲编码栏位，转成ndarray(栏位名称会遗失)
X = df[['color', 'size', 'price']].values
print(X)
# 先进行color 栏位标签编码，因为单热编码不能有object！(sklearn 0.22.2 以前旧版本的限制，0.22.2 以后(含)新版本无须先进行标签编码！)
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
# color 标签编码已完成
print(X)

## ------------------------------------------------------------------------
# 载入单热编码类别
from sklearn.preprocessing import OneHotEncoder
# 宣告类别物件ohe
#* ohe = OneHotEncoder(categorical_features=[0]) #  sklearn 0.22.2 以前的用法！
ohe = OneHotEncoder()
# 照预设编码完后转为常规矩阵
#* print(ohe.fit_transform(X).toarray())
print(ohe.fit_transform(df.iloc[:,[0,3]]).toarray()) # 单独挑选待编码栏位
print(np.hstack((ohe.fit_transform(df.iloc[:,[0,3]]).toarray(), df.iloc[:,1:3].values))) # 须将两numpy ndarray用值组 tuple 组织起来
# 或者可设定sparse 引数为False 传回常规矩阵
# ohe=OneHotEncoder(categorical_features=[0], sparse=False)
# print(ohe.fit_transform(X))

## ------------------------------------------------------------------------
# get_dummies() 编码前
print(df[['color', 'size', 'price']])
# pandas DataFrame 的get_dummies() 方法最为方便
print(pd.get_dummies(df[['color', 'size', 'price']])) # 预设drop_first=False是单热编码！
# print(pd.get_dummies(df[['color', 'size', 'price']], drop_first=True)) # drop_first=True才是虚拟编码！

#### 1.6 编程范式与物件导向概念
## ------------------------------------------------------------------------
# Python 泛函式编程语法示例
import numpy as np
# 用builtins 模组中的类别type 查核传入物件的类别
print(type([1,2,3]))
# 呼叫numpy 套件的std() 函数，输入为串列物件
print(np.std([1,2,3]))

## ------------------------------------------------------------------------
# Python 物件导向编程语法示例
# 以numpy 套件的array() 函数，将原生串列物件转换为衍生的
# ndarray 物件
a = np.array([1,2,3])
print(a)
print(type(a))
# 句点语法取用ndarray 物件a 的std() 方法
print(a.std())

## ------------------------------------------------------------------------
# Python 的numpy 套件向量化运算示例
print(np.sqrt(a))

## ------------------------------------------------------------------------
# 运用pandas 序列物件Series 之apply() 方法的隐式回圈
import pandas as pd
# 以pandas 套件的Series() 函数，将原生串列物件转换为衍生的
# Series 物件
a = pd.Series(a)
print(type(a))
# Python pandas 套件的apply() 方法
print(a.apply(lambda x: x+4))

#### 1.6.2 Python语言物件导向
## ------------------------------------------------------------------------
# 线性回归梯度陡降参数解法
# 定义类别LinearRegressionGD
class LinearRegressionGD(object):
    # 定义物件初始化方法，物件初始化时带有两个属性
    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter
    # 定义物件的方法fit()，此方法会根据传入的X 与y 计算属性
    # w_ 和cost_
    def fit(self, X, y):
        # 随机初始化属性w_
        self.w_ = np.random.randn(1 + X.shape[1]) # 加1是为了初始化截距项系数
        # 损失函数属性cost_
        self.cost_ = []
        # 根据物件属性eta 与n_iter，以及传入的(训练资料)X 与y
        # 计算属性 w_ 和cost_
        for i in range(self.n_iter):
            output = self.lin_comb(X) # 就是预测值y＾hat
            errors = (y - output) # y_i - y＾hat就是残差
            self.w_[1:] += self.eta * X.T.dot(errors) # Partial L/Partial b1
            self.w_[0] += self.eta * errors.sum() # Partial L/Partial b0
            cost = (errors**2).sum() / 2.0 # 式(1.1)
            self.cost_.append(cost)
        return self
    # 定义fit 方法会用到的lin_comb 线性组合方法
    def lin_comb(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    # 定义物件的方法predict()
    def predict(self, X):
        return self.lin_comb(X)

## ------------------------------------------------------------------------

# import LinearRegGD # 如果将类别定义函数独立为一个Python命令档
# dir(LinearRegGD)
        
# 前段程式码区块载入类别后，可发现环境中有LinearRegressionGD
# 此行叙述是Python 单列for 回圈写法，请参考1.8.2 节Python
# 语言资料汇入及汇出的串列推导(list comprehension)
print([name for name in dir() if name in
["LinearRegressionGD"]])
# 模拟五十笔预测变数，使用numpy 常用函数linspace()
X = np.linspace(0, 5, 50) # linspace(start, stop, num)
print(X[:4]) # 前四笔模拟的预测变数
# 模拟五十笔反应变数，利用numpy.random 模组从标准常态分布产生
# 随机乱数
y = 7.7 * X + 55 + np.random.randn(50)
print(y[:4])

## ------------------------------------------------------------------------
# 实作LinearRegressionGD 类物件lr
lr = LinearRegressionGD(n_iter=350)
# 创建后配适前有eta, n_iter, fit(), lin_comb() 与predict()
# print(dir(lr))
# 尚无w_ 与cost_ 属性
for tmp in ["w_", "cost_"]:
    print(tmp in dir(lr))
# 确认预设迭代次数已变更为350
print(lr.n_iter)
# 传入单行二维矩阵X 与一维向量y，以梯度陡降法计算系数
lr.fit(X.reshape(-1,1), y)
# 配适完毕后新增加w_ 与cost_ 属性
for tmp in ["w_", "cost_"]:
    print(tmp in dir(lr))
# 截距与斜率系数
print(lr.w_)
# 最后三代的损失函数值，随着代数增加而降低
print(lr.cost_[-3:])
# 预测X_new 的y 值
X_new = np.array([2])
print(lr.predict(X_new))
# X 与y 散布图及LinearRegressionGD 配适的线性回归直线
# Python 绘图语法参见4.1 节资料视觉化
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X, y)
ax.plot(X, lr.predict(X.reshape(-1,1)), color='red',
linewidth=2)
ax.set_xlabel('x')
ax.set_ylabel('y')
# fig.savefig('./_img/oo_scatterplot.png')

#### 1.8.2 Python语言资料汇入及汇出
## ------------------------------------------------------------------------
data_dir = "./_data/"
# Python 空字串的join() 方法，类似R 语言的paste0() 函数
fname = ''.join([data_dir, "letterdata.csv"])

## ------------------------------------------------------------------------
# mode 引数预设为'r' 读取模式
f = open(fname)
# 有read() 方法
print(dir(f)[49:54])
# read() 方法读档
data = f.read()
# 记得关闭档案连结
f.close()
# data 为str 类型物件
print(type(data))

## ------------------------------------------------------------------------
# 类别为str 的data 有712669 个字符
print(len(data))
# split() 方法依换行符号"\n" 将data 切成多个样本的lines
lines = data.split("\n")
# lines 类型为串列
print(type(lines))
# 检视第一列发现：一横列一元素，元素内逗号分隔开各栏位名称
# Python 串列取值冒号运算子，前包后不包
print(lines[0][:35])
# 再次以split() 方法依逗号切出首列中的各栏名称
header = lines[0].split(',')
print(header[:6])

## ------------------------------------------------------------------------
# 20002 笔
print(len(lines))
# 注意最末空字串
print(lines[20000:])
# 排除首列栏位名称与末列空字串
lines = lines[1:20001]

## ------------------------------------------------------------------------
# 第一笔观测值
print(lines[:1])
# 共两万笔观测值
print(len(lines))

## ------------------------------------------------------------------------
import numpy as np
# 宣告numpy 二维字符矩阵(20000, 17)
data = np.chararray((len(lines), len(header)))
print(data.shape)
# 以enumerate() 同时抓取观测值编号与观测值
for i, line in enumerate(lines):
    # 串列推导list comprehension，并入data 的第i 列
    data[i, :] = [x for x in line.split(',')]

## ------------------------------------------------------------------------
# 列印变数名称
# print(header)
# 列印各观测值
print(data)

## ------------------------------------------------------------------------
# 1.4.2 节pandas 读档指令
# fb = pd.read_excel("./_data/facebook_checkins_2013-08-24.xls"
# , skiprows = 1)
import pandas as pd
data_dir = "./_data/"
fname=''.join([data_dir,'/facebook_checkins_2013-08-24.xls'])
# 本节指定工作表名称与栏位名所在的横列数
fb = pd.read_excel(fname, sheet_name='总累积', header = 1)
# 读入后仍为pandas 套件DataFrame 物件
print(type(fb))

## ------------------------------------------------------------------------
# 检视前五笔数据
print(fb[['地标名称', '累积打卡数', '地区']].head())
# 纵向变数名称属性columns
print(fb.columns[:3])
# 横向观测值索引属性index(从0 到1000 间距1)
print(fb.index)

