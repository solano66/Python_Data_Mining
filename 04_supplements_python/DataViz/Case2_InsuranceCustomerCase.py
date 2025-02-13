#### **圖形文法繪圖(Grammar of Graphics Plotting, ggplot)**

#### 安裝plotnine(https://plotnine.readthedocs.io 容易安裝且維護良好！)

# !conda install conda-forge::plotnine --y (可能較慢！)
# !conda list plotnine

#### 安裝ggplot(https://github.com/yhat/ggpy 難搞且似乎維護不頻繁！建議創建ggplot虛擬環境後，在虛擬環境中安裝)
# 
# * linux系統
# 
#     - pip install numpy 
#     - pip install scipy 
#     - pip install statsmodels 
#     - pip install ggplot 
# 
# 
# * windows系統 
# 
#     下載ggplot安裝包(https://www.lfd.uci.edu/~gohlke/pythonlibs/#ggplot) 
# 
#     然後執行 
#     pip install ggplot‑0.11.5‑py2.py3‑none‑any.whl 
#     
#     或者是在Anaconda Prompt(請以系統管理員身份開啟)中輸入 conda install -c conda-forge ggplot


#### 快速入門
# ggplot通用語法如下：

# ggplot(data, mapping=aes(x=var1, y=var2)) + geom_plotname1() + geom_plotname2() + …… + stat_method1() + …… + additional_settings1 + …… 
# 資料集與底部圖層定義(座標及變數的對應) + 幾何繪圖物件1(通常是統計圖形名稱plotname1設為histogram) + 幾何繪圖物件2 + …… + 統計提煉方法1(通常是統計計算方法名稱lm) + 統計提煉方法2 + …… + 視覺美化額外設定1 + ……

#### Sample 1: meat資料集
# 套件載入
import plotnine as gp


# 如果執行時發生ImportError: cannot import name 'Timestamp'請跟著進行下列操作:
# 
# 找到 .../site-packages/ggplot/stats/smoothers.py 檔案並開啟，將
# 
# `from pandas.lib import Timestamp`
# 
# 改為
# 
# `from pandas import Timestamp`
# 
# 然後存檔，在試一次 `import ggplot as gp` 指令。

# 查看ggplot下含的物件
dir(gp)

# 載入並查看meat資料
from plotnine.data import meat
print(type(meat))
meat.head(10) # head(meat, 10) in R

meat.info()

# 產生一個以 meat 表格中 date 作為 X 軸 beef 作為 Y 軸的點圖
gp.ggplot(meat, gp.aes(x='date',y='beef')) + gp.geom_point(color='red') + gp.ggtitle('Scatter diagram of Date and Beef') + gp.theme(axis_text_x=gp.element_text(angle=30)) # data=meat挪前且刪除關鍵字
# axis.text.x in R ggplot2, but x_axis_text in Python ggplot, and axis_text_x in plotnine !!! dict(bottom=0.2) -> {'b': 0.2} -> drop it !

#### Sample 2: diamonds資料集
# 套件載入
# import ggplot as gp
import plotnine as gp1
import pandas as pd

# 鑽石資料集資料理解
# 
# - `diamonds` 資料集中包含了大約五萬多顆鑽石的資料，其中包含鑽石的 4C 品質指標，亦即顏色（color）、通透度（clarity）、切割等級（Cut）與重量克拉數（carat），另外還包含了幾個鑽石的尺寸資訊，詳細說明可參考 `diamonds` 的線上手冊。
from plotnine.data import diamonds

#d = diamonds
#print(d.head(10))
#print()
#print(d.info())
#print()
#print(d.describe(include='all'))

# 隨機抽取1000筆資料進行繪圖
# 
# - Reference: https://stackoverflow.com/questions/15923826/random-row-selection-in-pandas-dataframe

# Randomly sample 1000 elements from your dataframe
dsmall = diamonds.sample(n=1000)
dsmall

#### ggplot: 散點圖 - 1
# 
# - 底部圖層為兩量化變數price與carat分別對應到雙軸，因子變數color對應至顏色
# - 點圖層是尺寸大小為100的點繪圖
# - Reference : http://ggplot.yhathq.com/docs/geom_point.html

gp1.ggplot(dsmall, gp1.aes(x='carat', y='price', color='color')) + gp1.geom_point() # data=dsmall挪前且刪除關鍵字


#### ggplot: 散點圖 - 2
# 
# - 省略引數，無色彩變化
# - 線圖層欲繪製平滑曲線，`method = "lm"`表以線性迴歸方式配適直線，`se = False`表無信賴區間
# - 如果出現 AttributeError: 'DataFrame' object has no attribute 'sort' 請嘗試以下三種作法:
# 
#     1. 在E:/Anaconda3/Lib/site-packages/ggplot/stats/stat_smooth.py（當然這應該是ggplot包的位置）下改變77行：`smoothed_data = smoothed_data.sort('x')` 改為: `smoothed_data = smoothed_data.sort_values('x')` 。 
#     2. 在E:/Anaconda3/Lib/site-packages/ggplot/ggplot.py下改變602行：`fill_levels = self.data[[fillcol_raw, fillcol]].sort(=fillcol_raw)[fillcol].unique()` 改為：`fill_levels = self.data[[fillcol_raw, fillcol]].sort_values(by=fillcol_raw)[fillcol].unique()`
#     3. 如果前兩方式都不行的話，那我們還是將pandas的版本換為0.19.2版本的就行了：`pip install pandas==0.19.2`
#     
# 
# - Reference : 
#     * https://blog.csdn.net/llh_1178/article/details/79853850 
#     * https://stackoverflow.com/questions/50231134/how-to-downgrade-pandas-version

#### ggplot無法繪出迴歸直線，顯然在Python的實現有問題！
# gp.ggplot(dsmall, gp.aes('carat', 'price')) + gp.geom_point() + gp.stat_smooth(method="lm", se=False, color="blue")

gp1.ggplot(dsmall, gp1.aes('carat', 'price')) + gp1.geom_point() + gp1.stat_smooth(method="lm", se=False, color="blue")

#### ggplot: 散點圖 - 3
# 
# - 加上信賴區間

gp1.ggplot(dsmall, gp1.aes('carat', 'price')) + gp1.geom_point() + gp1.stat_smooth(method="lm", se=True, color="blue")

#### ggplot: 散點圖 - 4
# 
# - 底圖層以`colour = 'color'`將資料分組 (其實是多餘的！)
# - 點圖層再依color給各組不同顏色，點尺寸為5
# - 線圖層也依不同顏色繪製配適直線
# - Reference: https://mesfind.github.io/python-ecology-lesson/06-visualization-ggplot-python/

# gp1.ggplot(dsmall, gp1.aes('carat', 'price', colour='color')) + gp1.geom_point(colour='color', size=5) + gp1.stat_smooth(colour='color', method="lm", se=False) # stat_smooth圖層不能設colour!

#   File "/opt/anaconda3/lib/python3.7/site-packages/matplotlib/colors.py", line 233, in _to_rgba_no_colorcycle
#     raise ValueError("Invalid RGBA argument: {!r}".format(orig_c))

# ValueError: Invalid RGBA argument: 'color'

#### ggplot: 散點圖 - 5
# 
# - 最簡潔的寫法應該是在底圖層定義`color='color'`(`colour='color'`)，而非底圖層之`group = color`，及/或點線圖層加上`color = color` !
# - 重要觀念！前面圖層的設定，會影響到後面的圖層
# - 較簡潔的寫法如下：

gp1.ggplot(dsmall, gp1.aes('carat', 'price', colour='color')) + gp1.geom_point(size=5) +  gp1.stat_smooth(method="lm", se=False) # colour='color'依鑽石色澤分組，會影響後續的所有圖層


# 也可以這樣寫
p = gp1.ggplot(dsmall, gp1.aes('carat', 'price', colour = 'cut')) # 底部圖層定義，無任何繪圖動作！
p = p + gp1.geom_point() # 底部圖層加上點圖層
p

#### Sample 3: iris資料集
# #### ggplot: 線圖
# 
# - 三種鳶尾花共150株的iris資料集
# - 共四種特徵(sepal length, sepal width, petal length, petal width)
# - 有三種品種的鳶尾花(setosa, versicolor, virginica)
# 

from sklearn import datasets # conda install -n viz scikit-learn
import pandas as pd
iris = datasets.load_iris() # 載入鳶尾花資料集

print(iris.data)
print(type(iris.data))
print(iris.data.shape)
print(iris.target_names) 
print(iris.target.shape)
print(iris.feature_names) 
print(iris.target)

import numpy as np
X = pd.DataFrame(iris.data[:, :4],columns=['sepal length','sepal width','petal length','petal width'])
arrsp = iris.target
arrsp = np.where(arrsp==0, 'setosa', arrsp) # 替換 0 為花種字串 'setosa'
arrsp = np.where(arrsp=='1', 'versicolor', arrsp) # 注意！1和2已經變成字串型別了, 'versicolor'及'virginica'
arrsp = np.where(arrsp=='2', 'virginica', arrsp)
       
X['Species'] = arrsp # 添加類別標籤於X之後
X

gp1.ggplot(X, gp1.aes(x='petal length', y='petal width', color='Species')) + gp1.geom_line()

#### Case 2: Insurance Customer Case
import pandas as pd
custdata = pd.read_csv("./_data/custdata.tsv", sep='\t')
custdata.columns
custdata.rename(columns = {'is.employed': 'is_employed', 'marital.stat': 'marital_stat', 'health.ins': 'health_ins', 'housing.type': 'housing_type', 'recent.move': 'recent_move', 'num.vehicles': 'num_vehicles', 'state.of.res': 'state_of_res'}, inplace = True) # Python請善用句點！


#custdata.info()
custdata.describe(include="all")
#pd.set_option('display.max_columns', 500)
#pd.set_option('display.max_rows', 500)

#custdata.columns
#custdata['state_of_res'].head()
#custdata['state_of_res'].describe(include="all")
custdata['state_of_res'].value_counts()
custdata[custdata['housing_type'].isnull()].index
custdata[custdata['recent_move'].isnull()].index
custdata[custdata['num_vehicles'].isnull()].index
all(custdata[custdata['housing_type'].isnull()].index == custdata[custdata['recent_move'].isnull()].index)
all(custdata[custdata['recent_move'].isnull()].index == custdata[custdata['num_vehicles'].isnull()].index)

custdata.loc[:,('is_employed', 'housing_type', 'recent_move', 'num_vehicles')].describe(include="all")
#custdata.loc[:,('is_employed', 'housing_type', 'recent_move', 'num_vehicles')].sum()
custdata['income'].describe()
custdata['age'].describe()
custdata['income'].describe()
Income = custdata['income'] / 1000
Income.describe()

# from plotnine import * # have a cluttered environment
import plotnine as gp1

gp1.ggplot(custdata, gp1.aes(x='age')) + gp1.geom_histogram(binwidth=5, fill='gray') # data=custdata挪前且刪除關鍵字
#import matplotlib.pyplot as plt
#import seaborn as sns
#plt.hist(custdata["age"], bins=20, normed=True,color="gray")
#plt.xlabel("age")
#plt.ylabel("count")
#plt.show()

age_range = (custdata['age'].max() - custdata['age'].min()) # 全距 = 最大值 - 最小值
age_binwidth = age_range / 30 # 組距 = 全距 / 組數
print("合理組距：{}".format(age_binwidth))

# 連續版的直方圖(組距無限小，組數無限大下的直方圖)
gp1.ggplot(custdata, gp1.aes(x='age')) + gp1.geom_density() # data=custdata挪前且刪除關鍵字
#sns.distplot(custdata.age)
#plt.xlabel("age") 
#plt.ylabel("count")
#plt.show()

gp1.ggplot(custdata, gp1.aes(x='income')) + gp1.geom_density() # 連續版的直方圖(右偏分佈，$$$大多如此！長尾、冪律法則)

gp1.ggplot(data=custdata, mapping=gp1.aes(x='income')) + gp1.geom_density() + gp1.scale_x_log10(breaks=[100,1000,10000,100000])
#+annotation_logticks(sides="bt")

gp1.ggplot(data=custdata) + gp1.geom_density( mapping=gp1.aes(x='income')) + gp1.scale_x_log10(breaks=[100,1000,10000,100000])
#+annotation_logticks(sides="bt")

custdata.marital_stat.value_counts() # 四種婚姻狀態
gp1.ggplot(custdata) + gp1.geom_bar(gp1.aes(x='marital_stat'), fill="gray")

custdata.state_of_res.value_counts() # 50州
gp1.ggplot(custdata) + gp1.geom_bar(gp1.aes(x='state_of_res'), fill="gray") + gp1.coord_flip() # 根據原始數據進行統計後再繪圖，視覺壓力大！

# 將次數分佈表升冪排序，繪圖是由下往上繪製
statef = custdata['state_of_res'].value_counts()
statef = statef.reset_index(level=0)
# statef = statef.rename(columns={"index": "state_of_res", "state_of_res": "Count"})
statef = statef.sort_values(by=['count'])

# 排序後視覺壓力減輕！請注意stat="identity"，因為傳入的是統計表，而非原始數據！！！
gp1.ggplot(statef)+ gp1.geom_bar(gp1.aes(x='state_of_res', y='count'), stat="identity", fill="gray") + gp1.coord_flip() + gp1.theme(axis_text_y=gp1.element_text(size=5)) + gp1.scale_x_discrete(limits=statef.state_of_res)

custdata2 = custdata[(custdata.age > 0) & (custdata.age < 100) & (custdata.income > 0)]

gp1.ggplot(custdata2, gp1.aes(x='age', y='income')) + gp1.geom_point() + gp1.ylim(0, 200000)

gp1.ggplot(custdata2, gp1.aes(x='age', y='income')) + gp1.geom_point() + gp1.stat_smooth(method="lm") + gp1.ylim(0, 200000)

gp1.ggplot(custdata2, gp1.aes(x='age', y='income')) + gp1.geom_point() + gp1.geom_smooth(color='blue') + gp1.ylim(0, 200000)

# Initialsing Values 
# bool_val = [True, False] 
custdata2['healthinsnum'] = list(map(int, custdata2.loc[:,'health_ins']))

gp1.ggplot(custdata2, gp1.aes(x='age', y='healthinsnum')) + gp1.geom_point(position=gp1.position_jitter(0.05, 0.05)) + gp1.stat_smooth()

gp1.ggplot(custdata) + gp1.geom_bar(gp1.aes(x='marital_stat', fill='health_ins')) # 兩個類別變量的堆疊長條圖

gp1.ggplot(custdata) + gp1.geom_bar(gp1.aes(x='marital_stat', fill='health_ins'), position="dodge") # 兩個類別變量的並排長條圖

gp1.ggplot(custdata, gp1.aes(x='marital_stat')) + gp1.geom_bar(gp1.aes(fill='health_ins'), position="fill") + gp1.geom_point(gp1.aes(y=-0.05), size=0.75, alpha=0.3, position=gp1.position_jitter(0.01)) # 兩個類別變量的填滿式長條圖

gp1.ggplot(custdata2) + gp1.geom_bar(gp1.aes(x='housing_type', fill='marital_stat' ), position="dodge") + gp1.theme(axis_text_x=gp1.element_text(angle=45, hjust = 1))

gp1.ggplot(custdata2) + gp1.geom_bar(gp1.aes(x='housing_type', fill='marital_stat' ), position="dodge") + gp1.facet_wrap('housing_type', scales="fixed") + gp1.theme(axis_text_x=gp1.element_text(angle=45, hjust = 1))

gp1.ggplot(custdata2) + gp1.geom_bar(gp1.aes(x='housing_type', fill='marital_stat'), position="dodge") + gp1.facet_wrap('housing_type', scales="free") + gp1.theme(axis_text_x=gp1.element_text(angle=45, hjust = 1))
