'''
Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the IDS, CICD, and CISD of NTUB (臺北商業大學資訊與決策科學研究所暨智能控制與決策研究室教授兼校務永續發展中心主任); at the ME Dept. and CAIDS of MCUT (2020年借調至明志科技大學機械工程系擔任特聘教授兼人工智慧暨資料科學研究中心主任兩年); the CSQ (2019年起任中華民國品質學會大數據品質應用委員會主任委員); the DSBA (2013年創立臺灣資料科學與商業應用協會); and the CARS (2012年創立中華R軟體學會)
Notes: This code is provided without warranty.
'''

#### 4.1 資料視覺化
## ------------------------------------------------------------------------
# 載入必要套件，並記為簡要的名稱
import matplotlib.pyplot as plt
import numpy as np
# 產生或匯入資料
x = np.arange(0, 10, 0.2)
y = np.sin(x)
# 產生圖形(pyplot 語法)
plt.plot(x, y)
# 將圖形顯示在螢幕上
# plt.show()

# 載入必要套件，並記為簡要的名稱
import matplotlib.pyplot as plt
import numpy as np
# 產生或匯入資料
x = np.arange(0, 10, 0.2)
y = np.sin(x)
# 產生圖形(物件導向語法)
fig = plt.figure()
ax = fig.add_subplot(1,1,1) # (列，行，圖)
ax.plot(x, y)
# 將圖形顯示在螢幕上
# plt.show()
# 圖形儲存方法savefig()
# fig.savefig('./_img/plt.png', bbox_inches='tight')

# 載入必要套件pylab(過時！請跳過)
from pylab import *
# 產生或匯入資料
x = np.arange(0, 10, 0.2)
y = np.sin(x)
# 產生圖形(Matlab 語法)
plot(x, y)
# 將圖形顯示在螢幕上
# show()

# 載入必要套件，並記為簡要的名稱
import matplotlib.pyplot as plt
import numpy as np
# 產生或匯入資料
x = np.arange(0, 10, 0.2)
y = np.sin(x)
z = np.cos(x)
# 產生圖面與子圖(由此一次執行到plt.show()之前)
fig, axs = plt.subplots(nrows=2, ncols=1)
# 繪製第一個子圖正弦波，加上垂直軸標籤
axs[0].plot(x, y) # 高階繪圖
axs[0].set_ylabel('Sine') # 低階繪圖
# 繪製第二個子圖餘弦波，加上垂直軸說明文字
axs[1].plot(x, z) # 高階繪圖
axs[1].set_ylabel('Cosine') # 低階繪圖
# 將圖形顯示在螢幕上
# plt.show()
# 圖形儲存方法savefig()
# fig.savefig('./_img/multiplt.png', bbox_inches='tight')
# 還原圖形與子圖的預設設定
fig, ax = plt.subplots(nrows=1, ncols=1)

# 載入Python 語言pandas 套件與生化資料集
import pandas as pd
path = './_data/'
fname = 'segmentationOriginal.csv'
# 中間無任何空白的方式連結路徑與檔名
cell = pd.read_csv("".join([path, fname]))

# 119 個變數
print(len(cell.columns))

# 挑選五個量化變數
partialCell = cell[['AngleCh1', 'AreaCh1', 'AvgIntenCh1',
'AvgIntenCh2', 'AvgIntenCh3']]
# 以pandas 資料框的boxplot() (簡便繪圖)方法繪製並排盒鬚圖(圖4.3)
ax = partialCell.boxplot() # partialCell 是資料框物件
# pandas 圖形須以get_figure() 方法取出圖形後方能儲存
fig = ax.get_figure()
# fig.savefig('./_img/pd_boxplot.png')

# 以seaborn 套件的boxplot() 函數繪製並排盒鬚圖
import seaborn as sns
ax = sns.boxplot(x="variable", y="value",
data=pd.melt(partialCell)) # 10095 rows = 2019 samples * 5 variables

# 寬表轉長表自動生成的變數名稱variable 與value
print(pd.melt(partialCell)[2015:2022])

# seaborn 圖形也須以get_figure() 方法取出圖形後方能儲存
fig = ax.get_figure()
# fig.savefig('./_img/sns_boxplot.png')

# 載入Python 圖形文法繪圖套件及其內建資料集(建議用Python虛擬環境，搭配matplotlib 3.1.3版本，以及pandas 0.22.0版本，方能安裝ggplot與plotnine)
from ggplot import * # A messy environment gotten after such importing (!conda install -c conda-forge ggplot --y)

# 檢視鑽石資料集前5 筆樣本
import pandas as pd
print(diamonds.iloc[:, :9].head())

# 數值與類別變數混成的資料集
print(diamonds.dtypes)

# 圖形文法的圖層式繪圖
p = ggplot(aes(x='price', color='clarity'), data=diamonds) + geom_density() + scale_color_brewer(type='div') + facet_wrap('cut')
p

# ggplot 儲存圖形方法save()
p.save('./_img/gg_density.png')

#### ggplot繪圖部份建議執行下方程式碼，以使環境較不混亂！
# 載入Python 圖形文法繪圖套件及其內建資料集
# import ggplot as gp
import plotnine as gp # !conda install -c conda-forge plotnine --y
from plotnine.data import diamonds

# 檢視鑽石資料集前5 筆樣本
import pandas as pd
# print(gp.diamonds.iloc[:, :9].head())
print(diamonds.iloc[:, :9].head())

# 數值與類別變數混成的資料集
# print(gp.diamonds.dtypes)
print(diamonds.dtypes)

# 圖形文法的圖層式繪圖(底部圖層+密度曲線圖層+色盤控制圖層+分面繪製圖層)
# p = gp.ggplot(gp.aes(x='price', color='clarity'), data=gp.diamonds) + gp.geom_density() + gp.scale_color_brewer(type='div') + gp.facet_wrap('cut')
p = gp.ggplot(diamonds, gp.aes(x='price', color='clarity')) + gp.geom_density() + gp.scale_color_brewer(type='div') + gp.facet_wrap('cut')
p

# ggplot 儲存圖形方法save()
p.save('./_img/gg_density.png')
###

#### 4.2.2 線上音樂城關聯規則分析
## ------------------------------------------------------------------------
import pandas as pd
# 設定pandas 橫列與縱行結果呈現最大寬高值
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
# 線上音樂城聆聽記錄載入
lastfm = pd.read_csv("./_data/lastfm.csv")
# 聆聽歷程長資料
print(lastfm.head())

# 檢視欄位資料型別，大多是類別變數
print(lastfm.dtypes)

# 統計各用戶線上聆聽次數(一維頻次表)
print(lastfm.user.value_counts()[:5])

# 獨一無二的用戶編號長度，共有15000 位用戶
print(lastfm.user.unique().shape) # lastfm.user.nunique()

# 各藝人被點閱次數
print(lastfm.artist.value_counts()[:5])

# 確認演唱藝人人數，共有1004 位藝人
print(lastfm.artist.unique().shape) # lastfm.artist.nunique()

# 依用戶編號分組，grouped內有15,000個子表
grouped = lastfm.groupby('user')

# 檢視前兩組的子表，前兩位用戶各聆聽16 與29 位藝人專輯
print(list(grouped)[:2])

# 用戶編號有跳號現象
print(list(grouped.groups.keys())[:10])

# 以agg() 方法傳入字典，統計各使用者聆聽藝人數
numArt = grouped.agg({'artist': "count"})
print(numArt[5:10])

# 取出分組表藝人名稱一欄
grouped = grouped['artist']
# Python 串列推導，拆解分組資料為(巢狀nested或嵌套embedded)串列(長表 -> 寬表)
music = [list(artist) for (user, artist) in grouped]

# 限於頁面寬度，取出交易記錄長度<3 的數據呈現巢狀串列的整理結果
print([x for x in music if len(x) < 3][:2])

from mlxtend.preprocessing import TransactionEncoder # !conda install conda-forge::mlxtend --y
# pip install mlxtend --proxy="http://yourproxy:portnumber"
# 交易資料格式編碼(同樣是宣告空模 -> 配適實模-> 轉換運用)
te = TransactionEncoder()
# 傳回numpy 二元值矩陣txn_binary
txn_binary = te.fit(music).transform(music)
# 檢視交易記錄筆數與品項數
print(txn_binary.shape)

# 讀者自行執行dir()，可以發現te 實模物件下有columns_ 屬性
# dir(te)
# 檢視部分品項名稱
print(te.columns_[15:20])

# numpy 矩陣組織為二元值資料框(非常稀疏！False遠多於True)
df = pd.DataFrame(txn_binary, columns=te.columns_)
print(df.iloc[:5, 15:20])

# apriori 頻繁品項集(或強物項集)探勘(演算法)
from mlxtend.frequent_patterns import apriori
# pip install --trusted-host pypi.org mlxtend

# 挖掘時間長，因此記錄執行時間
# 可思考為何R 語言套件{arules} 的apriori() 快速許多？Ans. FORTRAN versus Python
import time
start = time.time()
freq_itemsets = apriori(df, min_support=0.01,
use_colnames=True)
end = time.time()
print(end - start)

# apply() 結合匿名函數統計品項集長度，並新增'length' 欄位於後
freq_itemsets['length'] = freq_itemsets['itemsets'].apply(lambda x: len(x))
# 頻繁品項集資料框，支持度、品項集與長度
print(freq_itemsets.head())

print(freq_itemsets.dtypes)

# 布林值索引篩選頻繁品項集
print(freq_itemsets[(freq_itemsets['length'] == 2)
& (freq_itemsets['support'] >= 0.05)])

# association_rules 關聯規則集生成
from mlxtend.frequent_patterns import association_rules
# 從頻繁品項集中產生49 條規則(生成規則confidence >= 0.5)
musicrules = association_rules(freq_itemsets,
metric="confidence", min_threshold=0.5)
print(musicrules.head())

# apply() 結合匿名函數統計各規則前提部長度
# 並新增'antecedent_len' 欄位於後
musicrules['antecedent_len'] = musicrules['antecedents'].apply(lambda x: len(x))
print(musicrules.head())

# 布林值索引篩選關聯規則
print(musicrules[(musicrules['antecedent_len'] > 0) &
(musicrules['confidence'] > 0.55)&(musicrules['lift'] > 5)])

#### 鐵達尼號資料集練習(補充)
import pandas as pd
titanic = pd.read_csv('./_data/Titanic.csv')
titanic = titanic.drop(['Unnamed: 0'], axis=1)
tf = pd.get_dummies(titanic) # 預設是單熱編碼

from mlxtend.frequent_patterns import apriori

freq_itemsets = apriori(tf, min_support=0.1, use_colnames=True)

from mlxtend.frequent_patterns import association_rules
# 從頻繁品項集中產生關聯規則(生成規則confidence >= 0.5)
titanicrules = association_rules(freq_itemsets,
metric="confidence", min_threshold=0.5)
# 關心結果部是frozenset({'Survived_Yes'})以及frozenset({'Survived_No'})的規則

#### 4.3.1 k平均數集群
## ------------------------------------------------------------------------
# library(animation) # Please try it in R!
# kmeans.ani() # Please try it in R!

#### 4.3.1.1 青少年市場區隔案例
## ------------------------------------------------------------------------
import numpy as np
import pandas as pd
teens = pd.read_csv("./_data/snsdata.csv")
# 文件詞項矩陣前面加上入學年、性別、年齡與朋友數等欄位
print(teens.shape) # 30000 * (4 + 36 terms by NLP)

# 留意gradyear 的資料型別
print(teens.dtypes)

# gradyear 更新為字串str 型別
teens['gradyear'] = teens['gradyear'].astype('str')
# 除了資料型別外，ftypes 還報導了屬性向量是稀疏還是稠密的
# print(teens.ftypes.head()) # deprecated after 1.0.0

# 各變數敘述統計值(報表過寬，只呈現部份結果)
print(teens.describe(include='all'))

# 各欄位遺缺值統計(只有gender 與age 有遺缺)(注意！此處並未進行遺缺值處理)
print(teens.isnull().sum().head())

# 各詞頻變數標準化建模
from sklearn.preprocessing import StandardScaler # Step 1

sc = StandardScaler() # Object-oriented programming paradigm, # Step 2

# 配適與轉換接續完成函數
teens_z = sc.fit_transform(teens.iloc[:,4:]) # Steps 3 & 4

# 錯誤用法！Python的類別函數不可使用泛函式編程語法
# teens_z = StandardScaler(teens.iloc[:,4:]) # > scikit-learn 0.23.2 可以！

# scikit-learn下preprocessing模組的scale()函數可用泛函式編程語法
from sklearn.preprocessing import scale
teens_z = scale(teens.iloc[:,4:])

# 資料導向程式設計經常輸出與輸入不同調(DataFrame 入ndarray 出)
print(type(teens_z))

# 轉為資料框物件取用describe() 方法確認標準化結果
print(pd.DataFrame(teens_z[:,30:33]).describe())
# mean 5.494864e-17  1.136868e-17 -9.687066e-17 可能數字上會有差異，這說明二進位制的計算機的數值運算不穩定性(numerical instability)

# Python k 平均數集群，隨機初始化的集群結果通常比較好
from sklearn.cluster import KMeans # Step 1
mdl = KMeans(n_clusters=5, init='random') # Step 2

# 訓練資料配適前空模的屬性與方法
pre = dir(mdl)
# 空模的幾個屬性與方法
print(pre[51:56])
# 以標準化文件詞項矩陣配適集群模型
import time
start = time.time()
mdl.fit(teens_z) # Step 3
end = time.time()
print("k-Means fitting spent {} seconds".format(end - start))

# 配適後實模的屬性與方法
post = dir(mdl)
# 實模的幾個屬性與方法
print(post[51:56])

# 實模與空模屬性和方法的差異(前或後有下底線_)
print(list(set(post) - set(pre)))

# sklearn 模型的存出(dump)與讀入(load)
import pickle
filename = './_data/kmeans.sav' # 設定存出路徑與檔名
pickle.dump(mdl, open(filename, 'wb')) # 'wb': write out in binary
res = pickle.load(open(filename, 'rb')) # 'rb': read in in binary
res = mdl
# res.labels_ 為30,000 名訓練樣本的歸群標籤
# import sys
# np.set_printoptions(threshold=sys.maxsize)
print(res.labels_.shape)

# 五群人數分佈(思考numpy下如何做！)
print(pd.Series(res.labels_).value_counts())

# 前10 個樣本的群編號
print (res.labels_[:10])

# 各群字詞平均詞頻矩陣的維度與維數
print(res.cluster_centers_.shape)

# 轉換成pandas 資料框，給予群編號與字詞名稱，方便結果詮釋
cen = pd.DataFrame(res.cluster_centers_, index = range(5),
columns = teens.iloc[:,4:].columns)
print(cen)

# 每次歸群結果的釋義會有不同
# Princesses: 3
# Criminals: 0
# Basket Cases: 4
# Athletes: 2
# Brains: 1

# 各群中心座標矩陣轉置後繪圖
ax = cen.T.plot() # seaborn, ggplot or pandas ?
# 低階繪圖設定x 軸刻度位置
ax.set_xticks(list(range(36)))
# 低階繪圖設定x 軸刻度說明文字
ax.set_xticklabels(list(cen.T.index), rotation=90)
fig = ax.get_figure()
fig.tight_layout()
# fig.savefig('./_img/sns_lineplot.png')

# 以下為課本/講義沒有的補充程式碼，主要在進行事後(建模後)的分析
# 添加群編號於原資料表後 eda + data preprocessing -> modeling -> post analysis
teens = pd.concat([teens, pd.Series(mdl.labels_).rename('cluster')], axis=1) # axis=1 means cbind() in R

# 抓集群未使用的三個變量(剛才歸群時未用，但事後分析確有助於了解各群的異同，以及歸群結果的品質)
teens[['gender','age','friends','cluster']][0:5]

# 各群平均年齡(群組與摘要也！)
teens.groupby('cluster').aggregate({'age': np.mean}) # 同儕間年齡差異不大！

# 新增是否為女生欄位'female'
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

# 各群女生人數比例(群組與摘要也！)
teens.groupby('cluster').aggregate({'female': np.mean})

teens.gender.value_counts()

# 各群朋友數(群組與摘要也！)
teens.groupby('cluster').aggregate({'friends': np.mean})

#### 4.3.3.1 密度集群案例
# 載入Python 密度集群類別DBSCAN()
from sklearn.cluster import DBSCAN # Density-Based Spatial Clustering of Applications with Noise
import numpy as np
import pandas as pd
# 讀取批發客戶資料集
data = pd.read_csv("./_data/wholesale_customers_data.csv")

# 注意各變數實際意義，而非只看表面上的數字
print(data.head())

print(data.dtypes)

# 移除名目尺度類別變數
data.drop(["Channel", "Region"], axis = 1, inplace = True)

# 二維空間可視覺化集群結果
data = data[["Grocery", "Milk"]]
# 集群前資料須標準化
#data = data.values.astype("float32", copy = False)
from sklearn.preprocessing import StandardScaler
stscaler = StandardScaler().fit(data)
data = stscaler.transform(data)
# 密度集群前，先繪製標準化後的樣本散佈圖
ax = pd.DataFrame(data, columns=["Grocery", "Milk"]).plot.scatter("Grocery", "Milk")
fig = ax.get_figure()
# fig.savefig('./_img/normalized_scatter.png')

# 以標準化資料配適DBSCAN 集群模型
dbsc = DBSCAN(eps = .5, min_samples = 15).fit(data)
# 歸群結果存出
labels = dbsc.labels_
# 雜訊樣本的群標籤為-1(numpy 產製次數分佈表的方式)
print(np.unique(labels, return_counts=True))

# # 設定繪圖顏色值陣列(書本上原繪圖程式碼)
# colors = np.array(['purple', 'blue'])
# # 利用labels+1 給定各樣本描點顏色
# ax = pd.DataFrame(data, columns=["Grocery", "Milk"]).plot.scatter("Grocery", "Milk", c=colors[labels+1])
# fig = ax.get_figure()
# # fig.savefig('./_img/dbscan_scatter.png')

#### 以下是繪圖加強版補充程式碼
# 密度集群結果資料框
import string
df = pd.DataFrame(data, columns=["Grocery", "Milk"])
df["Cluster"] = pd.Series(labels).apply(lambda x: string.ascii_lowercase[x+1])

# 設定繪圖點形、顏色與說明文字字典
marker_dict = {'a':'^', 'b':'o'} # '^': 三角點形; 'o': 圓形點形
color_dict = {'a':'blue', 'b':'red'}
label = {'a':'noise', 'b':'dense'}

# 依集群結果群組資料
groups = df.groupby(['Cluster']) # -1與0兩群
list(groups)
# 分組繪製不同點形與顏色的散佈圖
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.margins(0.05)
for name, group in groups:
    marker = marker_dict[name[0]]
    color = color_dict[name[0]]
    ax.plot(group.Grocery, group.Milk, marker=marker, linestyle='', label=label[name[0]], color=color) # ms=12, 
ax.legend()
ax.set_xlabel('Grocery')
ax.set_ylabel('Milk')
plt.show()
# fig.savefig('./_img/dbscan_scatter.png')
