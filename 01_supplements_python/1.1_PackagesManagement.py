'''
Collated by Ching-Shih Tsou 鄒慶士 博士 (Ph.D.) Distinguished Prof. at the Department of Mechanical Engineering/Director at the Center of Artificial Intelligence & Data Science (機械工程系特聘教授兼人工智慧暨資料科學研究中心主任), MCUT (明志科技大學); Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), NTUB (國立臺北商業大學)
Notes: This code is provided without warranty.
'''

### 1.1 套件管理

## Anaconda Navigator使用介紹與套件管理{#sec1.3:bg}

# 列出Python套件存放位置的命令提示字元或終端機模式指令
python -m site --user-site

## /Users/Vince/.local/lib/python3.8/site-packages

# What is python's site-packages directory? https://stackoverflow.com/questions/31384639/what-is-pythons-site-packages-directory

# 載入系統相關的參數與函數套件
import sys
# 列印搜尋模組的路徑集
print('\n'.join(sys.path))

## /opt/anaconda3/bin
## /opt/anaconda3/lib/python38.zip
## /opt/anaconda3/lib/python3.8
## /opt/anaconda3/lib/python3.8/lib-dynload
## /opt/anaconda3/lib/python3.8/site-packages
## /opt/anaconda3/lib/python3.8/site-packages/aeosa
## /Library/Frameworks/R.framework/Versions/3.5/Resources
## /library/reticulate/python

# 尋找指定套件安裝路徑
# https://myapollo.com.tw/2016/11/24/python-recipe-find-module-path/

# 先將套件pandas載入環境中
import pandas
# 運用套件sys下的modules方法查詢pandas安裝路徑
import sys
sys.modules['pandas']

## <module 'pandas' from '/opt/anaconda3/lib/python3.8
## /site-packages/pandas/__init__.py'>

# 或是更簡單的
import pandas
pandas.__path__

# 查詢已載入記憶體之Python套件或物件
dir()

## ['USArrests', '__annotations__', '__builtins__',
## '__doc__', '__loader__', '__name__', '__package__',
## '__spec__', 'fname1', 'fname2', 'pd', 'r', 'sys']

# 查看本機硬碟已安裝的Python套件(報表過長，請讀者自行執行)
!conda list
!pip list

# 查看本機已安裝套定套件的資訊
!conda list scikit-learn

## packages in environment at /opt/anaconda3:
## Name                    Version            
## scikit-learn              0.23.2
## Build  Channel
## py38h959d312_0

### 分析與繪圖{#sec1.3.1:bg}

# 載入Python套件sklearn內建資料集模組下的資料匯入方法
from sklearn.datasets import load_wine
#  有看到load_wine()
dir()
       
# 載入葡萄酒資料集
wine = load_wine()
type(wine)

# 載入並查看美國各州暴力犯罪率資料集USArrests前幾筆數據
# R語言預設是顯示前六筆；Python預設是前五筆
import pandas as pd
USArrests = pd.read_csv("./USArrests.csv", encoding='utf-8', index_col=0)
USArrests.head()

# 用sklearn中preprocessing模組的scale函數對USArrests進行標準化
from sklearn.preprocessing import scale
USArrests_z = scale(USArrests)

# 載入scipy階層式集群小套件cluster下的hierarchy模組
import scipy.cluster.hierarchy as sch
# sch.distance.pdist()函數計算兩兩州之間的歐基里德直線距離:
# 四維空間中成對樣本的距離(pairwise distances)
disMat = sch.distance.pdist(USArrests_z, 'euclidean')

# 依州間距離進行階層式集群
Z = sch.linkage(disMat, method='average')
# 集群結果儲存為numpy的n維陣列
type(Z)

# 載入Python重要繪圖小套件matplotlib.pyplot
# 將階層式集群結果以樹狀圖表示出來(繪圖程式碼需一起執行)
import matplotlib.pyplot as plt
# 宣告圖面尺寸
plt.figure(figsize = (10, 7))
# 繪製樹狀圖
P = sch.dendrogram(Z, labels = USArrests.index)
# 低階繪圖加上圖形標題
plt.title("Cluster Dendogram")

# 圖形物件P儲存為字典
type(P) # dict

# cut_tree指定群數(n_clusters)分群
res1 = sch.cut_tree(Z, n_clusters=[5])
# 檢視各群樣本數
import numpy as np
# 二維(50, 1)壓縮為一維(50,)
res1 = np.squeeze(res1)
# 各群樣本數量
np.unique(res1, return_counts=True)

# 取出位於Z第二列的樣本間距離值
# 找出最大值後乘以.7，作為分群的參考距離
res2 = sch.cut_tree(Z, height=np.max(Z[:,2])*0.7)
res2 = np.squeeze(res2)
# 結果碰巧與前面相同
np.unique(res2, return_counts=True)

### 資料視覺化{#sec1.3.2:bg}

# 載入資料集barley       
import pandas as pd
barley = pd.read_csv("./barley.csv", encoding = 'utf-8')
# info方法檢視資料集barley的結構
# 4個變數除了yield外其餘都是類別變數(R語言稱因子變數)
barley.info()

 # barley前六筆數據
barley.head(n = 6)

# 克里夫蘭點圖繪製，多維列聯表視覺化繪圖方法
import seaborn as sns
grid = sns.FacetGrid(barley, col='year', row='site')
# 給定橫軸與縱軸座標名稱
grid.map(plt.scatter, 'yield', 'variety')
