'''
Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the IDS, CICD, and CISD of NTUB (臺北商業大學資訊與決策科學研究所暨智能控制與決策研究室教授兼校務永續發展中心主任); at the ME Dept. and CAIDS of MCUT (2020年借調至明志科技大學機械工程系擔任特聘教授兼人工智慧暨資料科學研究中心主任兩年); the CSQ (2019年起任中華民國品質學會大數據品質應用委員會主任委員); the DSBA (2013年創立臺灣資料科學與商業應用協會); and the CARS (2012年創立中華R軟體學會) 
Notes: This code is provided without warranty.
'''

#### 1.5 向量化與隱式迴圈
# 方根函數應用到Python語言純量(傳統套用函數於單值的方式)
import math
a = 5
math.sqrt(a)

# math套件只能用來處理實數運算，而cmath套件可用來處理複數運算
import cmath
cmath.sqrt(a)

import numpy as np
num_sqrt2 = np.sqrt(a)
print(num_sqrt2)

#### 向量化(vectorization)
# 對每個元素進行四捨五入(vectorization)
import numpy as np
b = np.around([1.243, 5.654, 2.99])
print(b)


# 對數函數(vectorization)
in_array = np.random.rand(3,4)
print ("Input array : ", in_array)

out_array = np.log(in_array)
print ("Output array's log : ", out_array)


# 所有元素的平均值(axis=None, the default is to compute the mean of the flattened array.)
in_array = np.random.rand(3,4)
print ("Input array : ", in_array)

out_array = np.mean(in_array)
print ("Output array's mean : ", out_array)


# 各橫列平均值
in_array = np.random.rand(3,4)
print ("Input array : ", in_array)

out_array = np.mean(in_array, axis = 1)
print ("Output array's row mean : ", out_array)


# 各縱行平均值
in_array = np.random.rand(3,4)
print ("Input array : ", in_array)

out_array = np.mean(in_array, axis = 0)
print ("Output array's column mean : ", out_array)

#### Python對集合型物件中NA/nan的處理
# 建立含NA的陣列
in_array = np.random.rand(3,4)
in_array.ravel()[np.random.choice(in_array.size, replace=False)] = np.nan # 有趣的方法ravel()和屬性size，試著把ravel()拿掉
print ("Input array : ", in_array)


# 計算含NA的陣列平均值
out_array = np.mean(in_array)
print ("Output array's mean : ", out_array)

out_array = np.mean(in_array, axis = 0)
print ("Output array's column mean : ", out_array)

out_array = np.mean(in_array, axis = 1)
print ("Output array's row mean : ", out_array)

# np.nanmean(in_array, axis = 1)

#### Python的單row迴圈(單列迴圈)
# 創建三元素字典(最像R串列的Python資料物件)
d1 = {'x':[1,3,5], 'y':[2,4,6], 'z':['a', 'b']}
# 字典推導(dict comprehension)逐元素計算長度(or from toolz.dicttoolz import valmap)
d1_ld = {k: len(v) for k, v in d1.items()} # lapply() in R?
# 串列推導逐元素計算長度
d1_ll = [len(v) for v in d1.values()] # sapply() in R?

# 建立有向量和矩陣和的字典
# np.equal()判定對應元素是否相等
import numpy as np
import pandas as pd

firstList = {'A': np.arange(1,17).reshape(4,4), 'B': np.arange(1,17).reshape(2,8), 'C': np.arange(1,6)}
secondList = {'A': np.arange(1,17).reshape(4,4), 'B': np.arange(1,17).reshape(8,2), 'C': np.arange(1,16)}

{k: np.array_equal(v, secondList[k]) for k, v in firstList.items()} # The keys in firstList and secondList are the same

#### 自定義函數
def simpleFunc(x, y): 
    x_len = x.shape[1] if x.ndim > 1 else len(x)
    y_len = y.shape[1] if y.ndim > 1 else len(y)
    return x_len + y_len

#### 結合單row迴圈
{k: simpleFunc(v, secondList[k]) for k, v in firstList.items()}

#### iris分組散佈加迴歸直線圖
iris = pd.read_csv('iris.csv', encoding='utf-8')

import seaborn as sns
sns.set(color_codes=True) # context or styles
g = sns.lmplot(x="Petal.Length", y="Sepal.Width", hue="Species", data=iris, markers=["o", "x" ,"+"])


#### 以下為書本上沒有的範例(因為講義中是R範例)：Apply-like methods/functions in Pandas and Numpy
import pandas as pd
import numpy as np
# 運用字典創建資料框 Create a data frame from a dict
data = {'name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'], 
        'year': [2012, 2012, 2013, 2014, 2014], 
        'reports': [4, 24, 31, 2, 3],
        'coverage': [25, 94, 57, 62, 70]}
type(data)

df = pd.DataFrame(data, index = ['Cochice', 'Pima', 'Santa Cruz', 'Maricopa', 'Yuma'])
df
type(df)
# 有名字的匿名函數(anonymous function) ~ 有點怪！
capitalizer = lambda x: x.upper() # 轉大寫

#### 1D apply
df['name'].apply(capitalizer)
'Amy'.upper()

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

data = {'name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'], 
        'year': [2012, 2012, 2013, 2014, 2014], 
        'reports': [4, 24, 31, 2, 3],
        'coverage': [25, 94, 57, 62, 70]}
type(data)

df = pd.DataFrame(data, index = ['Cochice', 'Pima', 'Santa Cruz', 'Maricopa', 'Yuma'])

df.applymap(times100)


#### apply() applies a function along axes in the dataframe.
# Drop the string variable so that applymap() can run
df = df.drop('name', axis=1)
df.apply(np.std, axis=0) # column apply

df.apply(np.mean, axis=1) # row apply

#### numpy.apply_along_axis: Apply a function to 1-D slices along the given axis.
import numpy as np

def partial_avg(v):
    """頭尾值的平均 Average first and last element of a 1-D array"""
    return (v[0] + v[-1]) * 0.5

b = np.array([[1,2,3], [4,5,6], [7,8,9]])

b

np.apply_along_axis(partial_avg, axis=0, arr=b) # column apply

np.apply_along_axis(partial_avg, 1, b) # row apply


b = np.array([[8,1,7], [4,3,9], [5,2,6]])
b
np.apply_along_axis(sorted, 1, b) # row apply

# For a function that returns a higher dimensional array, those dimensions are inserted in place of the axis dimension.
# 另一個有趣的例子，先創建二維陣列
b = np.array([[1,2,3], [4,5,6], [7,8,9]])
b

# 沿著b的最後一軸套用對角矩陣函數後，再將結果疊成三維陣列
np.apply_along_axis(np.diag, axis=-1, arr=b)
# 結果同上
np.apply_along_axis(np.diag, 1, b)
# 沿著b的第一軸套用的結果相當耐人尋味，試著從另外一個方向看！
np.apply_along_axis(np.diag, 0, b) # 須從另外一個方向看！

#### References:
# https://chrisalbon.com/python/data_wrangling/pandas_apply_operations_to_dataframes/
# https://numpy.org/doc/stable/reference/generated/numpy.apply_along_axis.html
