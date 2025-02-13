'''
Collated by Prof. Ching-Shih Tsou 邹庆士 教授 (Ph.D.) at the IDS, CICD, and CISD of NTUB (台北商业大学资讯与决策科学研究所暨智能控制与决策研究室教授兼校务永续发展中心主任); at the ME Dept. and CAIDS of MCUT (2020年借调至明志科技大学机械工程系担任特聘教授兼人工智慧暨资料科学研究中心主任两年); the CSQ (2019年起任品质学会大数据品质应用委员会主任委员); the DSBA (2013年创立台湾资料科学与商业应用协会); and the CARS (2012年创立中华R软体学会) 
Notes: This code is provided without warranty.
Dataset: iris.csv
'''

#### 1.5 向量化与隐式回圈
# 方根函数应用到Python语言纯量
import math
a = 5
math.sqrt(a)

# math套件只能用来处理实数运算，而cmath套件可用来处理复数运算
import cmath
cmath.sqrt(a)

import numpy as np
num_sqrt2 = np.sqrt(a)
print(num_sqrt2)

#### 向量化(vectorization)
# 对每个元素进行四舍五入(vectorization)
import numpy as np
b = np.around([1.243, 5.654, 2.99])
print(b)


# 对数函数(vectorization)
in_array = np.random.rand(3,4)
print ("Input array : ", in_array)

out_array = np.log(in_array)
print ("Output array's log : ", out_array)


# 所有元素的平均值(axis=None, the default is to compute the mean of the flattened array.)
in_array = np.random.rand(3,4)
print ("Input array : ", in_array)

out_array = np.mean(in_array)
print ("Output array's mean : ", out_array)


# 各横列平均值
in_array = np.random.rand(3,4)
print ("Input array : ", in_array)

out_array = np.mean(in_array, axis = 1)
print ("Output array's row mean : ", out_array)


# 各纵行平均值
in_array = np.random.rand(3,4)
print ("Input array : ", in_array)

out_array = np.mean(in_array, axis = 0)
print ("Output array's column mean : ", out_array)

#### Python对集合型物件中NA/nan的处理
# 建立含NA的阵列
in_array = np.random.rand(3,4)
in_array.ravel()[np.random.choice(in_array.size, replace=False)] = np.nan # 有趣的方法ravel()和属性size，试着把ravel()拿掉
print ("Input array : ", in_array)


# 计算含NA的阵列平均值
out_array = np.mean(in_array)
print ("Output array's mean : ", out_array)

out_array = np.mean(in_array, axis = 0)
print ("Output array's column mean : ", out_array)

out_array = np.mean(in_array, axis = 1)
print ("Output array's row mean : ", out_array)

out_array = np.nanmean(in_array, axis = 1)
print ("Output array's row mean : ", out_array)

#### Python的单row回圈
# 创建三元素字典(最像R串列的Python资料物件)
d1 = {'a':[1,3,5], 'b':[2,4,6], 'c':['a', 'b']}
# 字典推导逐元素计算长度(or from toolz.dicttoolz import valmap)
d1_ld = {k: len(v) for k, v in d1.items()} # lapply() in R?
# 串列推导逐元素计算长度
d1_ll = [len(v) for v in d1.values()] # sapply() in R?

# 建立有向量和矩阵和的字典
# np.equal()判定对应元素是否相等
import numpy as np
import pandas as pd

firstList = {'A': np.arange(1,17).reshape(4,4), 'B': np.arange(1,17).reshape(2,8), 'C': np.arange(1,6)}
secondList = {'A': np.arange(1,17).reshape(4,4), 'B': np.arange(1,17).reshape(8,2), 'C': np.arange(1,16)}

{k: np.array_equal(v, secondList[k]) for k, v in firstList.items()} # The keys in firstList and secondList are the same

#### 自定义函数
def simpleFunc(x, y): 
    x_len = x.shape[1] if x.ndim > 1 else len(x)
    y_len = y.shape[1] if y.ndim > 1 else len(y)
    return x_len + y_len

#### 结合单row回圈
{k: simpleFunc(v, secondList[k]) for k, v in firstList.items()}

#### iris分组散布加回归直线图
iris = pd.read_csv('iris.csv', encoding='utf-8')

import seaborn as sns
sns.set(color_codes=True) # context or styles
g = sns.lmplot(x="Petal.Length", y="Sepal.Width", hue="Species", data=iris, markers=["o", "x" ,"+"])


#### 以下为书本上没有的范例：Apply-like methods/functions in Pandas and Numpy
import pandas as pd
import numpy as np
# 运用字典创建资料框 Create a data frame from a dict
data = {'name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'], 
        'year': [2012, 2012, 2013, 2014, 2014], 
        'reports': [4, 24, 31, 2, 3],
        'coverage': [25, 94, 57, 62, 70]}
df = pd.DataFrame(data, index = ['Cochice', 'Pima', 'Santa Cruz', 'Maricopa', 'Yuma'])
df
type(df)
# 有名字的匿名函数(anonymous function) ~ 有点怪！
capitalizer = lambda x: x.upper()

#### 1D apply
df['name'].apply(capitalizer)
df['name'].apply(lambda x: x.upper())

#### Or you can use map() to do the vectorization
# map() applies an operation over each element of a series
df['name'].map(capitalizer)

#### 2D applymap
# applymap() applies a function to every single element in the entire dataframe.
# Drop the string variable so that applymap() can run
df = df.drop('name', axis=1)

# Return the square root of **every cell** in the dataframe
df.applymap(np.sqrt)

def times100(x):
    # that, if x is a string,
    if type(x) is str:
        # just returns it untouched
        return x
    # but, if not, return it multiplied by 100
    else:
        return 100 * x


df.applymap(times100)


#### apply() applies a function along axes in the dataframe.
df.apply(np.mean, axis=0) # column apply, colmeans() in R

df.apply(np.mean, axis=1) # row apply, rowmeans() in R

#### numpy.apply_along_axis: Apply a function to 1-D slices along the given axis.
import numpy as np

def my_func(a):
    """头尾值的平均 Average first and last element of a 1-D array"""
    return (a[0] + a[-1]) * 0.5

b = np.array([[1,2,3], [4,5,6], [7,8,9]])

b

np.apply_along_axis(my_func, axis=0, arr=b) # column apply

np.apply_along_axis(my_func, 1, b) # row apply


b = np.array([[8,1,7], [4,3,9], [5,2,6]])
b
np.apply_along_axis(sorted, 1, b) # row apply

#### References:
# https://chrisalbon.com/python/data_wrangling/pandas_apply_operations_to_dataframes/
# https://numpy.org/doc/stable/reference/generated/numpy.apply_along_axis.html
