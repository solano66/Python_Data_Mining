# -*- coding: utf-8 -*-
'''
Collated by Ching-Shih Tsou 鄒慶士 博士 (Ph.D.) Distinguished Prof. at the Department of Mechanical Engineering/Director at the Center of Artificial Intelligence & Data Science (機械工程系特聘教授兼人工智慧暨資料科學研究中心主任), MCUT (明志科技大學); Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), NTUB (國立臺北商業大學)
Notes: This code is provided without warranty.
'''

#### 3.5.1 數值變數與順序尺度類別變數
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

iris = pd.read_csv('iris.csv', index_col=[0], encoding='utf-8')
iris.columns

 # 鳶尾花資料集量化變數成對散佈圖
import seaborn as sns
sns.pairplot(iris, hue='Species') # hue='Species'不同花種用不同的顏色

# 載入美國2006年新生兒資料集，資料請取出WTGAIN 與DBWT 均無遺缺值的樣本子集，並鎖定單胞胎與移除早產兒樣本
births2006_smpl = pd.read_csv('births2006.smpl.csv', index_col=[0], encoding='utf-8') # 427323 * 13
births2006_smpl.isnull().sum()

# 取出WTGAIN與DBWT均無遺缺值的樣本子集
births2006_cln = births2006_smpl.dropna(axis=0, subset=['WTGAIN', 'DBWT'])

# 鎖定單胞胎與移除早產兒樣本
births2006_cln = births2006_cln.loc[(births2006_cln['DPLURAL'] == "1 Single") & (births2006_cln['ESTGEST'] > 35),:]
births2006_cln.shape

# 繪製母親懷孕體重增加值與嬰兒體重關係圖
plt.scatter(births2006_cln["WTGAIN"], births2006_cln["DBWT"])

# 共變異數(方陣)
births2006_cln.loc[:,["WTGAIN","DBWT"]].cov()
(10*births2006_cln.loc[:,["WTGAIN","DBWT"]]).cov()

# 皮爾森相關係數(方陣)
births2006_cln.loc[:,["WTGAIN","DBWT"]].corr()

np.corrcoef(births2006_cln["WTGAIN"], births2006_cln["DBWT"])

# 史皮爾曼相關係數(方陣)
births2006_cln.loc[:,["WTGAIN","DBWT"]].corr('spearman')

# 模擬十筆5 個變量的資料矩陣
X = np.random.randint(1, 7, size=[10,5])
# 共變異數方陣
np.cov(X)
np.cov(X).shape # (10, 10), numpy.cov()以橫向為變量方向

# 載入sklearn共變異數穩健統計類別方法
# 最小共變異數判別式估計法，support_fraction即為前述之h
from sklearn.covariance import MinCovDet
cov = MinCovDet(support_fraction=0.75,random_state=0).fit(X)
cov.covariance_ 
cov.covariance_.shape # (5, 5), sklearn.covariance以縱向為變量方向
cov.location_

#### 3.5.2 名目尺度類別變數
# 關節炎案例形式類別資料
Arthritis = pd.read_csv('Arthritis.csv', encoding='utf-8')

Arthritis.head()
Arthritis.info()

# 除了ID與Age之外，其餘均為字串變數
Arthritis.dtypes

#  觀測值總數
Arthritis.shape[0]

#  變數個數
Arthritis.shape[1]

# 頻次形式類別資料
# 自訂expand.grid() 建立sex 與party 的所有可能組合及其頻次
def expand_grid(data_dict):
    rows = itertools.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())

GSS = expand_grid({'sex': ['Male', 'Female'],'party': ['dem', 'indep', 'rep']})
GSS['count']=[279,165,73,47,225,191]

#  所有因子水準組合下的頻次形式
print(GSS)

#  字串變數與頻次構成的表格
GSS.info()

#  總觀測值個數
sum(GSS['count'])

#  各因子所有水準的組合數
print(GSS.shape[0])

# 載入鐵達尼號原始資料
titanic_raw = pd.read_csv('titanic.raw.csv', index_col=[0], encoding='utf-8')
titanic_raw.info()

# titanic_raw.select_dtypes('object').apply(lambda x: x.value_counts(), axis=0)
for i in range(4):
    print('Frequency table of {}:'.format(titanic_raw.columns[i]))
    print(titanic_raw.iloc[:,i].value_counts())
    print()

titanic_freq = pd.crosstab([titanic_raw['Class'],titanic_raw['Sex'],titanic_raw['Age']], titanic_raw['Survived'], margins=True)

#  觀測值總數
titanic_freq.iloc[-1,:]

#### 3.5.3 類別變數視覺化關聯檢驗
HairEyeColor = pd.read_csv('HairEyeColor.csv',encoding='utf-8')
print(HairEyeColor[HairEyeColor.loc[:,'Sex']=='Male'].pivot(index='Eye', columns='Hair',values='Freq'))
print(HairEyeColor[HairEyeColor.loc[:,'Sex']=='Female'].pivot(index='Eye', columns='Hair',values='Freq'))

HairEyeColor.loc[:,'Freq'].sum()

HairEyeColor[HairEyeColor.loc[:,'Sex']=='Male'].pivot(index='Eye', columns='Hair',values='Freq').add(HairEyeColor[HairEyeColor.loc[:,'Sex']=='Female'].pivot(index='Eye', columns='Hair',values='Freq'))

UCBAdmissions = pd.read_csv('UCBAdmissions.csv',encoding='utf-8')

A = UCBAdmissions[UCBAdmissions.loc[:,'Dept']=='A'].pivot(index='Gender',columns='Admit',values='Freq')
B = UCBAdmissions[UCBAdmissions.loc[:,'Dept']=='B'].pivot(index='Gender',columns='Admit',values='Freq')
C = UCBAdmissions[UCBAdmissions.loc[:,'Dept']=='C'].pivot(index='Gender',columns='Admit',values='Freq')
D = UCBAdmissions[UCBAdmissions.loc[:,'Dept']=='D'].pivot(index='Gender',columns='Admit',values='Freq')
E = UCBAdmissions[UCBAdmissions.loc[:,'Dept']=='E'].pivot(index='Gender',columns='Admit',values='Freq')
F = UCBAdmissions[UCBAdmissions.loc[:,'Dept']=='F'].pivot(index='Gender',columns='Admit',values='Freq')
x = A.add(B).add(C).add(D).add(E).add(F)


# import numpy as np
# import matplotlib.pyplot as plt

# # radius of each bar
# radii = x.to_numpy().flatten() 

# # Value - width
# width = np.pi/ 2

# # angle of each bar
# theta = np.radians([0,90,180,270])

# ax = plt.subplot(111, polar=True)
# bars = ax.bar(theta, radii, width=width, alpha=0.5)
# plt.show()
