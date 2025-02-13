########################################################## Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the IDS and CADS(資訊與決策科學研究所暨資料科學應用研究中心), NTUB(國立臺北商業大學); the CARS(中華R軟體學會); and the DSBA(臺灣資料科學與商業應用協會)
#######################################################
### Notes: This code is provided without warranty.

# Practice OO
class Account:
    def __init__(self, name, number, balance):
        self.name = name
        self.number = number
        self.balance = balance
 
    def deposit(self, amount):
        if amount <= 0:
             raise ValueError('amount must be positive')
        self.balance += amount
 
    def withdraw(self, amount):
        if amount > self.balance:
            raise RuntimeError('balance not enough')
        self.balance -= amount
 
    def __str__(self):
        return 'Account({0}, {1}, {2})'.format(
            self.name, self.number, self.balance)


acct = Account('Justin', '123-4567', 1000)
acct
dir(acct)
print(acct)
acct.deposit(500) # Try -500
print(acct)
acct.withdraw(200) # Try 2000
print(acct)

### Practice 1 ###
# https://unicode-table.com/cn/2639/
cry = u"\u2639"
cry

ord(cry) # Return the Unicode code point for a one-character string.

help(ord)

len(cry)

cry.encode('utf8')

type(cry.encode('utf8')) # bytes

len(cry.encode('utf8')) # bytes長度為3

# print(u"\u2639".encode('ascii'))

## Practices 3
#list0 = [168, 'Price', ['Industry1', 'Industrty2']]
#list0[0]
#list0[1]
#list0[2][1]

# Practice 4


# Practice 5
list1 = [1, 2, 3]
list2 = [4, 5, 6]
list1 + list2

# Practice 6
import numpy as np
mat1 = np.array(list1)
mat2 = np.array(list2)
mat1
mat1 + mat2 # vectorization

# Practice 7
mat1.dot(mat2)

# Practice 8
#correlation
import pandas as pd
TRD_Index=pd.read_table('Basics/data/TRD_Index.txt',sep='\t')
TRD_Index.Indexcd.value_counts()
TRD_Index.head() # SH: 1, SZ: 399106
SHindex=TRD_Index[TRD_Index.Indexcd==1] # logical indexing
SHindex.head(3)

SZindex=TRD_Index[TRD_Index.Indexcd==399106]
SZindex.head(3)

#import matplotlib.pylot as plt # ModuleNotFoundError: No module named 'matplotlib.pylot'
from matplotlib import pyplot as plt
plt.scatter(SHindex.Retindex,SZindex.Retindex)
plt.title('散佈圖')
plt.xlabel('上海綜合證券指數收益率')
plt.ylabel('深圳綜合證券指數收益率')

#SZindex.index=SHindex.index
#SZindex.Retindex.corr(SHindex.Retindex)

### Practice
import pandas as pd
from collections import OrderedDict
from datetime import date

# 左上：row-centered dict 橫導向的字典(串列中包著字典)
sales = [{'account': 'Jones LLC', 'Jan': 150, 'Feb': 200, 'Mar': 140},
         {'account': 'Alpha Co',  'Jan': 200, 'Feb': 210, 'Mar': 215},
         {'account': 'Blue Inc',  'Jan': 50,  'Feb': 90,  'Mar': 95 }]
df = pd.DataFrame(sales)

# 左下：column-centered dict 縱導向的字典(字典中有串列)
sales = {'account': ['Jones LLC', 'Alpha Co', 'Blue Inc'],
         'Jan': [150, 200, 50],
         'Feb': [200, 210, 90],
         'Mar': [140, 215, 95]}
df = pd.DataFrame(sales)
# 或者是
# df = pd.DataFrame.from_dict(sales)

# OrderedDict和縱導向的字典是一樣的意思！
sales = OrderedDict([ ('account', ['Jones LLC', 'Alpha Co', 'Blue Inc']),
          ('Jan', [150, 200, 50]),
          ('Feb',  [200, 210, 90]),
          ('Mar', [140, 215, 95]) ] )
df = pd.DataFrame(sales)
# df = pd.DataFrame.from_dict(sales)


# 右上：row-centered list 橫導向的串列(串列中包著值組)
sales = [('Jones LLC', 150, 200, 50),
         ('Alpha Co', 200, 210, 90),
         ('Blue Inc', 140, 215, 95)]
# df = pd.DataFrame(sales)
# 要給變數名稱
labels = ['account', 'Jan', 'Feb', 'Mar']
df = pd.DataFrame(sales, columns=labels)
# 或者是
# df = pd.DataFrame.from_records(sales, columns=labels)

# 右下：column-centered list 縱導向的串列(串列中包著值組)
sales = [('account', ['Jones LLC', 'Alpha Co', 'Blue Inc']),
         ('Jan', [150, 200, 50]),
         ('Feb', [200, 210, 90]),
         ('Mar', [140, 215, 95])]
df = pd.DataFrame(sales) # 不是我們要的樣子！
# df = pd.DataFrame.from_items(sales) # from_items方法已經過時！
# AttributeError: type object 'DataFrame' has no attribute 'from_items'

# Practice 9
import pandas as pd
print (pd.__version__)

df = pd.DataFrame({'Gender':['f', 'f', 'm', 'f', 'f', 'm', 'm', 'f', 'm', 'f', 'm'], 'TV': [3.4, 3.5, 2.6, 4.7, 4.2, 4.2, 5.1, 3.9, 3.7, 2.1, 4.3]})

df
df.dtypes

df['Gender'][3:7] # 3, 4, 5, 6
df[['TV', 'Gender']][3:7] # 3, 4, 5, 6
df[3:7][['TV', 'Gender']] # 3, 4, 5, 6
help(df.iloc)
df.iloc[3:7, 0:2] # 連續位置取值 3, 4, 5, 6 & 0, 1

df.iloc[[3,5], 0:2] # 間斷位置取值 3, 5 & 0, 1

df.loc[3:7, 'Gender'] # 3, 4, 5, 6, 7*
df.loc[3:7, ['TV']] # 3, 4, 5, 6, 7* with variable name 'TV'

dir(df)
grouped = df.groupby('Gender')
grouped
list(grouped)
type(grouped)
dir(grouped)

grouped.describe()
help(grouped.get_group)
# grouped.get_group("M")
grouped.get_group("m")

grouped.boxplot()

# Practice 10

import pandas as pd
nba = pd.read_csv("./data/nba_2013.csv")

nba.dtypes
type(nba)

nba.describe(include = "all")
help(nba.describe)

nba.head() # It's a wide data !

# Wide to long
nba_l = nba.stack()

# Long to wide
nba_l.unstack()

### Practice 11

#by pandas
nba.columns
nba.sort_values(by="fg")

### Practice 12
import statsmodels.api as sm
airquality = sm.datasets.get_rdataset("airquality")



type(airquality) # statsmodels.datasets.utils.Dataset

airquality = pd.DataFrame(airquality['data'])

airquality.isnull()
airquality.isnull().sum(axis=0)
airquality.isnull().sum(axis=1)



