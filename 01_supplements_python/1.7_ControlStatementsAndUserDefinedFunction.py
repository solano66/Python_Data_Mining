'''
Collated by Ching-Shih Tsou 鄒慶士 博士 (Ph.D.) Distinguished Prof. at the Department of Mechanical Engineering/Director at the Center of Artificial Intelligence & Data Science (機械工程系特聘教授兼人工智慧暨資料科學研究中心主任), MCUT (明志科技大學); Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), NTUB (國立臺北商業大學)
Notes: This code is provided without warranty.
'''

#### 1.7.1 R語言控制敘述範例
# 運用for迴圈對串列中的元素一一進行平方運算
x = [5, 12, 13]
# 迴圈敘述關鍵字for、: 及下方內縮
for n in x:
  print(n**2)


# while迴圈
i = 1
# 迴圈敘述關鍵字while
while (i <= 10):
   i = i + 4

print(i)


# 運用while迴圈進行牛頓法求根
# 解的初始值
x = 2
# 欲求根的函數
f = x ** 3 + 2 * x ** 2 - 7
# 牛頓法容許誤差
tolerance = 0.00001
while (abs(f) > tolerance):
   # 求根函數的一階導函數
   f_prime = 3 * x **2 + 4 * x
   # 以牛頓法的根逼近公式更新解
   x = x - f / f_prime
   # 新解的函數值
   f = x ** 3 + 2 * x ** 2 - 7
# 印出解
print(x)

# 運用repeat迴圈進行牛頓法求根
x = 2
tolerance = 0.000001
while True:
  f = x**3 + 2 * x**2 - 7
  if abs(f) < tolerance:
    break
  f_prime = 3 * x**2 + 4 * x
  x = x - f/f_prime
print(x)

# (補充範例)條件判斷敘述if...else...
grade = ["C", "e", "d", "B", "F"]
for i in range(len(grade)):
    if grade[i].isupper():
        print(grade[i] + " is uppercase.")
    else:
        print(grade[i] + " is not uppercase.")

# 條件判斷敘述if...else...
import pandas as pd
import numpy as np
grade = ["C", "C-", "A-", "B", "F"]
#print(type(grade))
check = np.repeat(True, 5)
for i in range(len(grade)):
  #print(i.isalpha())
  #print(type(i))
  #print(isinstance(i, str))
  if not isinstance(grade[i], str):
    check[i] = False
    
if any(check):
  grade = pd.Series(grade, dtype="category")

grade.dtypes

# 再以if not grade.dtypes == "category"判定grade是否為類別向量，結果如為TRUE則執行if關鍵字下方程式碼一次；結果如為FALSE則執行else關鍵字下方程式碼一次，印出條件判定的結果說明：`"Grade already is a factor."`。
if not grade.dtypes == "category":
  grade = pd.Series(grade, dtype="category")
else:
  print("Grade already is a category.")

#### 1.7.2 R語言自訂函數範例
import matplotlib.pyplot as plt
import numpy as np
# 注意Python語言自訂函數，注意關鍵字def        ，以及三個引數在函數主體如何運用
def corplot(x, y, plotit = False):
  if plotit == True:
    plt.plot(x, y,'o')
    #plt.plot(x,y)
    plt.show()
  return np.corrcoef(x,y)

#def corplot(x, y, plotit = False):
#    if plotit == True:
#        plt.figure('Draw')
#        plt.scatter(x, y)
#        print(np.corrcoef(x, y))
#    else:
#        print(np.corrcoef(x, y))

# 隨機產生u,v亂數
u = np.random.uniform(2,8,(10,))
v = np.random.uniform(2,8,(10,))  
#u = np.random.randint(2, 8, size=(1,10))
#v = np.random.randint(2, 8, size=(1,10))

# 函數呼叫與傳入引數u與v
corplot(u, v)
# 改變plotit默認值
corplot(u, v, True)

##### 1.7.2 R語言自訂函數新生資料集案例
'''
Some comparisons between R and Python Pandas
R               | Pandas
--------------------------------------
summary(df)            | df.describe()
head(df)               | df.head()
dim(df)                | df.shape
dplyr::slice(df, 1:10) | df.iloc[:9]
'''
import numpy as np
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

df = pd.read_excel('106_freshmen_final-toR_language.xls', sheet_name='106 新生資料 ')
print(df.head(5))
#print(df.iloc[:5])
print(df.info())
#  字串變數次數統計
print(df.describe(include='all'))

temp = df.describe(include='all')

#def rstr(df): return df.shape, df.apply(lambda x: [x.unique()], axis=1)
#print(rstr(df))

#  性別變數有異常
print(df['性別'].describe())
print(df['性別'].unique())
print(df['性別'].value_counts())
df['性別'].replace({'1男':'男','2女':'女'}, inplace=True)

print(df['性別'].unique())
print(df['性別'].value_counts())

df.dtypes

#  將選定欄位成批產生次數分佈表
sel = df.iloc[:,[4,5,6,7,8]]
# 法一
sel.apply(lambda x: x.value_counts(), axis=0).T.stack() # 有幾個 x ? 五個 x

# 法二
for col in sel.columns:
  print(col, ":")
  freq = df[col].value_counts()
  idxFreq = [freq.index.tolist(), freq.values.tolist()] # [['女', '男'], [1775, 733]]
  for lst in idxFreq:
    for element in lst:
      print(element, end='\t')
    print()
  print()

# 法三: 自己的寫法

# 系科學制自訂函數設計
def deptByAcaSys(dept="企管系", acasys="四技"):
  filter1 = df['系所'] == dept # 2508 True/False
  filter2 = df['學制'] == acasys # 2508 True/False
  tbl = df[filter1 & filter2] # 企管四技
  #tbl = tbl[filter2]
  top3 = tbl.groupby(['畢業學校'])['班級代碼'].count().reset_index(name='count').sort_values(['count'], ascending=False).head(3)
  bottom3 = tbl.groupby(['畢業學校'])['班級代碼'].count().reset_index(name='count').sort_values(['count'], ascending=False).tail(3)
  newdf = pd.concat([top3.reset_index(drop=True), bottom3.reset_index(drop=True)], axis=1)
  newdf.columns = ["Top","TopFreq","Bottom","BottomFreq"]
  return newdf

# 照預設值呼叫函數，仍然要加上成對小括弧
print(deptByAcaSys())
# 改變函數預設值
print(deptByAcaSys("會資系","二技"))

##### (補充) control statements
x = -99
if x < 0:
    print ('It is negative')


x = 99
if x < 0:
    print ('It is negative')
elif x == 0:
    print ('Equal to zero')
elif 0 < x < 5:
    print ('Positive but smaller than 5')
else:
    print ('Positive and larger than or equal to 5')


x=0
while x < 5:
    print (x, "is less than 5")
    x += 1


for x in range(5):
    print (x, "is less than 5")


for x in range(10):
    if x==3:
        continue # go immediately to the next iteration
    if x==5:
        break # quit the loop entirely
    print (x)


##### (補充)User-defined functions
def my_function(x, y, z=1.5): # positional arguments and keyword arguments 位置引數與關鍵字引數
    if z > 1:
        return z * (x + y)
    else:
        return z / (x + y)


my_function(5, 6, z=0.7)


my_function(3.14, 7, 3.5)


def func():
    a = []
    for i in range(5):
        a.append(i)


func() # return nothing because of local scoping
# a # NameError: name 'a' is not defined


a = []
def func():
    for i in range(5):
        a.append(i)


func()
a


a


##### (補充：小心使用控制敘述) Logical/Boolean Indexing & Control Statements
##### Example 1
##### Old school 老派作法
import numpy as np
help(np.random.uniform)

scores = np.random.uniform(low=0, high=100, size=10) # 人性本懶
grades = np.chararray(10, itemsize=5) # character array
# grades = "\0" * 10 # NOT right !

for i in range(len(scores)):
    if (scores[i] < 60):
        grades[i] = "Fail"
    else:
        grades[i] = "Pass"

scores
grades.decode('utf8')

##### A suggested way 建議寫法
scores = np.random.uniform(0, 100, 10)
grades = np.chararray(10)
grades[scores < 60] = r"Fail" # 60 broadcasting (cycling in R) + vectorization
grades
grades[scores >= 60] = r"Pass"
grades

##### Example 2
import numpy as np
scores = np.random.uniform(0, 100, 10)
ages = np.random.uniform(40, 70, 10)

# Old school
for i in range(len(scores)):
    if ages[i] >= 60:
        scores[i] = scores[i] + 10

scores

# A suggested way
scores[ages >=60] = scores[ages >= 60] + 10
scores



#################################################
#### 以下暫時勿理
#import pandas as pd
#
#df = pd.read_excel ('./106_freshmen_final-toR_language.xls')
#
#
#df.dtypes
#
#
#df['入學學年'] = df['入學學年'].astype('category')
#df['系所代碼'] = df['系所代碼'].astype('category')
#df['班級代碼'] = df['班級代碼'].astype('category')
#df['班級名稱'] = df['班級名稱'].astype('category')
#df['部別'] = df['部別'].astype('category')
#df['學制'] = df['學制'].astype('category')
#df['系所'] = df['系所'].astype('category')
#df['學院'] = df['學院'].astype('category')
#df['性別'] = df['性別'].astype('category')
#df['畢業學校'] = df['畢業學校'].astype('category')
#
#print(df)
#print(df.head())
#print(df.describe())
#print(df.columns)
#
#sex = pd.crosstab(index=df["性別"], columns="count")
#print(sex)
#school = pd.crosstab(index=df["畢業學校"], columns="count")
#
#
#
#import pandas as pd
#import numpy as np
#import functools as ft
#
#def main():
#    # Create dataframe
#    df = pd.DataFrame(data=np.zeros((0, 3)), columns=['word','gender','source'])
#    df["word"] = ('banana', 'banana', 'elephant', 'mouse', 'mouse', 'elephant', 'banana', 'mouse', 'mouse', 'elephant', 'ostrich', 'ostrich')
#    df["gender"] = ('a', 'the', 'the', 'a', 'the', 'the', 'a', 'the', 'a', 'a', 'a', 'the')
#    df["source"] = ('BE', 'BE', 'BE', 'NL', 'NL', 'NL', 'FR', 'FR', 'FR', 'FR', 'FR', 'FR')
#
#    return create_frequency_list(df)
#
#def create_frequency_list(df):
#    xtabs = df.groupby(df.columns.tolist()).size() \
#              .unstack([2, 1]).sort_index(1).fillna(0).astype(int)
#
#    total = xtabs.stack().sum(1)
#    total.name = 'total'
#    total = total.to_frame().unstack()
#
#    return pd.concat([total, xtabs], axis=1)
#
#main()
#
#def main():
#    # Create dataframe
#    df_test = pd.DataFrame(data=np.zeros((0, 3)), columns=['sex','school'])
#    df_test['sex'] = sex['count']
#    df_test['school'] = sex['count']
#
#    return create_frequency_list(df_test)
#
#def create_frequency_list(df):
#    xtabs = df.groupby(df.columns.tolist()).size() \
#              .unstack([2, 1]).sort_index(1).fillna(0).astype(int)
#
#    total = xtabs.stack().sum(1)
#    total.name = 'total'
#    total = total.to_frame().unstack()
#
#    return pd.concat([total, xtabs], axis=1)
#
#main()