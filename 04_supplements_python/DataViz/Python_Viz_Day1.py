'''
Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the IDS, CICD, CISD, and CCE of NTUB (臺北商業大學資訊與決策科學研究所暨智能控制與決策研究室教授兼校務永續發展中心主任&推廣教育部主任); at the ME Dept. and CAIDS of MCUT (2020年借調至明志科技大學機械工程系擔任特聘教授兼人工智慧暨資料科學研究中心主任兩年); the CSQ (2019年起任中華民國品質學會AI暨大數據品質應用委員會主任委員); the DSBA (2013年創立臺灣資料科學與商業應用協會); and the CARS (2012年創立中華R軟體學會)
Datasets: 唐詩三百首.csv, poem.csv, custdata.tsv, lvr_landAcsv/A_LVR_LAND_A.CSV, 2330.csv, y1s600000.xls, 空氣品質監測站/空氣品質監測站位置圖_121_10704.shp, 空氣品質監測站_10704.json, Dengue_Daily.csv, parrot.png
Notes: This code is provided without warranty.
'''

#### **Matplotlib基礎繪圖**
# 可參考 https://lijin-thu.github.io/06.%20matplotlib/06.01%20pyplot%20tutorial.html
from matplotlib import pyplot as plt
import numpy as np

#### 折線圖
price = [100, 200, 300, 400, 500, 600, 700]
vol = [945, 857, 642, 341, 239, 100, 32]
plt.plot(price, vol) # x, y, fmt (a format string), please try plt.plot(price, vol, 'bo') and plt.plot(price, vol, '-o')
plt.show()

#### 長條圖(陽春版)
category = ['A', 'B', 'C', 'D', 'E']
scores = [30, 63, 10, 88, 92]
plt.bar(category, scores)

plt.show()

# 橫向長條圖(陽春版)
plt.barh(category, scores) # barh: horizontal bar plot
plt.show()

#### 添加圖形標題與xy軸標籤(進階版)
plt.barh(category, scores)
plt.title('Category vs Scores')
plt.xlabel('Scores')
plt.ylabel('Category')
plt.show()

#### 選用繪圖形式
# site-packages\matplotlib\mpl-data\stylelib
plt.style.use('ggplot') 
plt.barh(category, scores)
plt.show()

#### 直方圖
seq = np.random.randn(500)
plt.hist(seq)
plt.show()

# 建立新的繪圖物件
fig = plt.figure()
fig

fig = plt.figure(figsize = (6, 6)) # width, height in inches.
plt.plot(price) # missing x
plt.show()

#### 添加子圖法一：plt -> figure -> add_subplot -> ax -> 繪各子圖 ax.xxxx() 或 plt.xxxx()
fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)
fig
plt.show()


fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
plt.plot(price, vol)
ax2 = fig.add_subplot(2, 2, 2) # fig有前綴add_，且用ax?接
ax3 = fig.add_subplot(2, 2, 3)
plt.barh(category, scores)
plt.xlabel('Scores')
plt.ylabel('Category')
ax4 = fig.add_subplot(2, 2, 4)
plt.show()

# One more example
#### 添加子圖法二：plt -> figure, plt -> subplot
# - 分割繪圖區域為2*3網格，並選定網格1，繪製線圖
# - 選擇網格2，繪製柱狀圖
# - 選擇網格3，繪製橫向柱狀圖
# - 選擇網格4，繪製堆疊式柱狀圖
# - 選擇網格5，繪製盒鬚圖
# - 選擇網格6，繪製散佈圖

x = [1,2,3,4]
y = [5,4,3,2]
plt.figure()
plt.subplot(231)
plt.plot(x, y) # Plot y versus x as lines and/or markers. x-y散佈圖
plt.subplot(232)
plt.bar(x, y) # x-y長條圖
plt.subplot(233)
plt.barh(x, y) # x-y水平長條圖
plt.subplot(234)
plt.bar(x, y) # x-y長條圖
y1 = [7,8,5,3]
plt.bar(x, y1, bottom=y, color = 'b') # 疊加長條圖，將紅色y1疊加在y之上
plt.subplot(235)
plt.boxplot(x) # x盒鬚圖
plt.subplot(236)
plt.scatter(x,y) # x-y散佈圖，另一個繪圖函數
# 沒有add_前綴，直接plt一直畫下去，plt.figure(), plt.subplot(), plt.xxxx()

#### Exercise 1: 繪製生化資料集'EqSphereAreaCh1'與'PerimCh1'的散佈圖，以及BC轉換後的散佈狀況並，以'Class'為散點的顏色。
import pandas as pd
cell = pd.read_csv('./data/segmentationOriginal.csv')
from sklearn.preprocessing import LabelEncoder
cl = LabelEncoder().fit_transform(cell.Class)
import matplotlib
import matplotlib.pyplot as plt
colors = ['purple', 'blue']
plt.scatter(cell.EqSphereAreaCh1, cell.PerimCh1, c=cl, cmap=matplotlib.colors.ListedColormap(colors))
plt.show()

#### BC轉換後的散佈圖
from scipy import stats

plt.scatter(stats.boxcox(cell.EqSphereAreaCh1)[0], stats.boxcox(cell.PerimCh1)[0], c=cl, cmap=matplotlib.colors.ListedColormap(colors))

# plt.xlim(stats.boxcox(cell.EqSphereAreaCh1)[0].min(), stats.boxcox(cell.EqSphereAreaCh1)[0].max())

# plt.ylim(stats.boxcox(cell.PerimCh1)[0].min(), stats.boxcox(cell.PerimCh1)[0].max())

plt.show()

help(plt.scatter) # 後不贅述

#### 線條樣式與顏色
fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(price, vol, linestyle='-', color='g') # 減
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(price, vol, linestyle='--', color='red') # 減減
ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(price, vol, linestyle='-.', color='blue') # 減點
ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(price, vol, linestyle=':', color='purple') # 冒號
plt.show()

#### 注意colspan和rowspan兩參數
plt.figure()
# row上往下，col左往右，等份型
ax1 = plt.subplot2grid((2,3), (0,0), colspan = 3, rowspan=1) # (2,3)表分two rows three columns，(0,0)表從1st row 1st column開始繪圖
ax1.plot(price, vol, linestyle='-', color='g')
ax2 = plt.subplot2grid((2,3), (1,0), colspan = 2, rowspan=1) # (2,3)表分two rows three columns，(1,0)表從2nd row 1st column開始繪圖
ax2.plot(price, vol, linestyle='--', color='red')
ax3 = plt.subplot2grid((2,3), (1,2), colspan = 1, rowspan=1) # (2,3)表分two rows three columns，(1,2)表從2nd row 3rd column開始繪圖
ax3.plot(price, vol, linestyle='-.', color='blue')
plt.show()


#### 添加圖例
x1 = np.random.rand(50).cumsum()
x2 = np.random.rand(50).cumsum()

print(x1)

plt.plot(x1, linestyle='--', label = 'x1')
plt.plot(x2, linestyle=':', label = 'x2')
plt.legend(loc='best')
# best
# upper right/left/center
# lower right/left/center
# center right/left
# center


#### 設置座標軸上下界
plt.plot(x1, linestyle='--', label = 'x1')
plt.plot(x2, linestyle=':', label = 'x2')
plt.xlim((20, 40)) # 限縮x軸範圍
plt.legend(loc='best')
plt.show()


#### 設置軸座標刻度
plt.plot(x1, linestyle='--', label = 'x1')
plt.plot(x2, linestyle=':', label = 'x2')
plt.xlim((20, 40))
plt.xticks(np.linspace(20, 40 ,5)) # (20, 25, 30, 35, 40) also make it ! 前包後也包
plt.legend(loc='best')
plt.show()


#### 隱藏座標軸
plt.plot(x1)
plt.xticks(()) # 傳入空值組 empty tuple
plt.yticks(()) # 傳入空值組 empty tuple
plt.show()


#### 設置左右雙座標軸
fig, ax = plt.subplots() # 注意！plt.subplots()有兩個回傳值: fig和ax

ax1 = ax.twinx()

ax.plot(price, color='red') # 左垂直軸
ax1.plot(vol, color='blue') # 右垂直軸

plt.show()

#### 佈局設定
#### 圖形嵌入
fig = plt.figure()
# 原點在左下角
# 注意左下寬高的比例值，0到1的範圍
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax = fig.add_axes([left, bottom, width, height])
ax.plot(x1)

left1, bottom1, width1, height1 = 0.2, 0.6, 0.2, 0.2
ax1 = fig.add_axes([left1, bottom1, width1, height1])
ax1.plot(x2, color = 'blue') # 左上圖

left1, bottom1, width1, height1 = 0.6, 0.2, 0.2, 0.2
plt.axes([left1, bottom1, width1, height1])
plt.plot(x2, color = 'blue') # 右下圖

plt.show()

#### 添加註釋
plt.style.use('classic') 

ts = [1,3,4,6,8,9,23,18,12,5]

x0 = 6
y0 = 23

plt.plot(ts)

plt.annotate('max value is 23', 
             xy = (x0, y0), # 箭頭指向位置
             fontsize = 10,
             arrowprops = dict(arrowstyle='->', connectionstyle='arc3, rad=.1'), # Please try rad=.5
             xytext = (+3, +12) # text起始位置
            )

#### 長條圖添加注釋
plt.bar(category, scores)
# 注意zip()的聰明用法與轉字串函數str()
for x, y, text in zip(range(len(category)), scores, scores):
    plt.text(x = x, y = y + 1.4, s = str(text), ha = 'center', va = 'center')

plt.show()

#### 散佈圖
height = [150, 168, 159, 144, 136, 158, 166]
weight = [59, 49, 78, 69, 65, 77, 45]
plt.scatter(height, weight)
plt.show()

#### 儲存圖片
# plt.bar(category, scores)

# for x, y, text in zip(range(len(category)), scores, scores):
#     plt.text(x = x, y = y + 1.4, s = str(text), ha = 'center', va = 'center')

# plt.savefig('barplot')

#### Case 1: 鳶尾花資料集案例(matplotlib + pandas + seaborn)
from matplotlib import pyplot as plt
# plt.rcParams['figure.figsize'] = [7, 5]
import numpy as np
import pandas as pd
import seaborn as sns
# import warnings
# warnings.filterwarnings('ignore')
df = sns.load_dataset('iris')
df.head()

type(df)

# > matplotlib線圖(line plot)

# - 花萼長度的折線圖
plt.style.use('ggplot') # 使用ggplot繪圖形式
plt.plot(df.sepal_length) # 橫軸為樣本編號


# - Setosa的花萼長度折線面積圖
sl = df.sepal_length.values # numpy ndarray
print(len(sl)) # 150
plt.plot(sl[0:50], linestyle='-.')
plt.fill_between(range(50), sl[0:50],
                 color="skyblue", alpha=0.4)


# - 花萼長度疊加圖，不同線形(個別取值繪圖三次)
plt.plot(sl[0:50], linestyle='-.', label='setosa')
plt.plot(sl[50:100], linestyle='--', label='versicolor')
plt.plot(sl[100:150],linestyle=':', label='virginica')
plt.legend(loc='best')

# > matplotlib堆疊圖(stack plot)

# - y軸範圍與上圖不同，因為對應樣本值已逐漸累計加總
# - labels引數加註說明文字標籤
# - plt.legend()低階繪圖函數加上圖例說明於左上角
x = range(50)
plt.stackplot(x, # not x*3 !!!
              sl[0:50], 
              sl[50:100], 
              sl[100:150], 
              labels=['setosa','versicolor','virginica'])
plt.legend(loc='upper left')


# > 群組與摘要

# - 資料理解與探索的前哨工作
# - 以類別變數將資料分組後進行特徵的統計計算
g = df.groupby('species').mean()
g


# > matplotlib並排長條圖(side by side barplot)
ind = np.asarray([0, 1, 2]) # the x locations for the groups Setosa, Versicolor, and Virginica
# tranpose DataFrame g and transforme it to Numpy ndarray
value = g.T.values
p1 = plt.bar(ind, value[0], width=0.2, label='Sepal Length') # the width of the bars
p2 = plt.bar(ind+0.2, value[1], width=0.2, label='Sepal Width') # 注意x方向右移與寬度均為0.2
p3 = plt.bar(ind+0.4, value[2], width=0.2, label='Petal Length')
p4 = plt.bar(ind+0.6, value[3], width=0.2, label='Petal Width')


for p in [p1, p2, p3, p4]: # 四組長條圖
    for i in p: # 各組三長條
        height = i.get_height() # 抓長條的高度
        plt.text(
            i.get_x() + i.get_width()/2., # 抓長條的寬度起點，並算橫軸座標
            height*1.05, # 稍微放大長條的高度
            '%s' % str(round(height, 2)), # 置入高度數字
            ha = "center", # horizontal alignment
            va = "bottom", # vertical alignment
        )

plt.ylabel("")
plt.xlabel("Species")
plt.title("Iris")
plt.xticks(ind+0.2 ,("Setosa","Versicolor","Virginica"))
plt.legend(loc='upper left',fontsize='xx-small')

# 了解代碼
value[0] # the first row of value

value.shape

ind

# > pandas DataFrame長條圖
# 
# - 直接用群組與摘要的結果表pandas DataFrame下的plot()方法，繪製並排長條圖(side by side barplot)
# - 注意行導向繪圖
# - pandas.DataFrame.plot, pandas.DataFrame.plot.area, pandas.DataFrame.plot.bar, pandas.DataFrame.plot.barh, pandas.DataFrame.plot.box, pandas.DataFrame.plot.density, pandas.DataFrame.plot.hexbin, pandas.DataFrame.plot.hist, pandas.DataFrame.plot.kde, pandas.DataFrame.plot.line, pandas.DataFrame.plot.pie, pandas.DataFrame.plot.scatter, pandas.DataFrame.boxplot, pandas.DataFrame.hist
type(g)
# dir(g)
# dir(g.plot)
g.plot.bar() # 挺方便的！

# - 填滿式長條圖(filled barplot)
g2 = g/g.sum() * 100 # Try g.max()
type(g2)

g2

# - 以stacked引數，結合上表繪出填滿式長條圖
g2.T.plot.bar(stacked=True)

# > matplotlib盒鬚圖(box-and-whisker plot)
plt.boxplot(df.petal_length)
plt.show()


# > seaborn盒鬚圖
df.species
df.petal_length

# sns.set(style="ticks", palette="pastel")
sns.boxplot(x=df.species, y=df.petal_length)


# > seaborn盒鬚圖加上晃動後的散點
sns.boxplot(x=df.species, y=df.petal_length)
sns.swarmplot(df.species, y=df.petal_length, color="grey")


# - 調整盒子寬度與線寬
sns.boxplot(df.species, df.petal_length, width=0.5, linewidth=2.5)


# - order引數可以改變盒子呈現順序
sns.boxplot(df.species, df.petal_length, width=0.5, linewidth=2.5,
            order=['virginica', 'setosa', 'versicolor'])


# - 盒鬚圖註記
ax = sns.boxplot(df.species, df.petal_length)
median = df.groupby(['species'])['petal_length'].median()
print(median)
species_num = df.species.value_counts().values
print(species_num)
for i in range(len(species_num)):
    ax.text(i, median[i], str(species_num[i]), # 在中位數線上加註樣本數
            size='large', color='w')


#### 另一個例子(小費資料集)
# 載入資料集
tips = sns.load_dataset("tips")
tips

type(tips)
tips.day.value_counts()

# 繪圖 
sns.boxplot(x="day", y="total_bill", hue="sex", data=tips, palette="PRGn") 
sns.despine(offset=10, trim=True)


# 回到鳶尾花資料集
# > matplotlib直方圖
plt.hist(df.petal_length, bins=9)


#### seaborn distplot
# 
# - This function combines the matplotlib hist function (with automatic calculation of a good default bin size) with the seaborn kdeplot() and rugplot() functions. It can also fit scipy.stats distributions and plot the estimated PDF over the data.
# - 雙峰分佈
sns.distplot(df.petal_length, bins=20)



sns.distplot(df.petal_length, bins=10,
             vertical=True,
             kde=True,
             kde_kws={'color': 'y', 'alpha':0.1, 
                      'linewidth': 1, 'shade':True}) # 密度曲線下方有陰影


#### matplotlib分割圖面，seaborn繪圖
fig, (ax1, ax2) = plt.subplots(2, sharex=True, # x方向共軸
                               gridspec_kw={"height_ratios": (0.2, 0.8)}) # y方向分配比例

sns.boxplot(df.sepal_width, ax=ax1) # 輸出ax1圖形
sns.distplot(df.sepal_width, ax=ax2) # 輸出ax2圖形


# - seaborn預設後圖與前圖疊加於同一個圖面(此種圖形稱為疊加圖，有別於分面圖)
sns.distplot(df.sepal_length, color="green", label='Sepal Length')
sns.distplot(df.sepal_width, color="darkred", label='Sepal Width')
plt.legend()


#### seaborn散佈圖加迴歸線(regplot)
# 
# - Plot data and a linear regression model fit.
sns.regplot(x=df.petal_width, y = df.sepal_width, fit_reg=True)



sns.regplot(x=df.petal_width, y = df.sepal_width, fit_reg=False)


#### seaborn散佈加分組迴歸線圖(lmplot)
# 
# - Plot data and regression model fits across a FacetGrid.

# 疊加圖 versus 分面圖

# hue以因子變數設定顏色(color in R)，markers設定繪圖字符形狀(pch/plotting character in R)
sns.lmplot(x='sepal_width', y='sepal_length', data = df, fit_reg=True, 
           hue='species', markers=["x", "o", "+"]) 
plt.legend()

# col='species'
sns.lmplot( x='sepal_width', y='sepal_length', data = df, fit_reg=True, 
           col='species')

sns.lmplot(x='sepal_width', y='sepal_length', data = df, fit_reg=False, 
           hue='species', legend=False, markers=["x", "o", "+"]) # try legend=True
plt.legend()


#### seaborn雙變量散佈圖與單變量直方圖(jointplot)
# 
# - 雙變量散佈圖(判關係)加上單變量直方圖(看分佈) Draw a plot of two variables with bivariate and univariate graphs.
sns.jointplot(x=df.petal_length, y=df.sepal_length, kind='scatter')


#### 散佈圖改為hexbin plot(2D直方圖，建議資料點比較多時使用)
sns.jointplot(x=df.petal_length, y=df.sepal_length, kind='hex')


#### 一、二維核函數密度估計圖
sns.jointplot(x=df.petal_length, y=df.sepal_length, kind='kde')


#### seaborn成對散佈圖(pairplot)
# ValueError: Filled and line art markers cannot be mixed
# https://matplotlib.org/stable/gallery/lines_bars_and_markers/marker_reference.html
sns.pairplot(df, kind="scatter", hue="species", markers=["v", "o", "D"], palette="Set2")


#### matplotlib氣泡圖(bubble plot)
# 
# - 利用散佈圖繪製函數產生氣泡圖
# - 橫軸變量、縱軸變量、氣泡變量(s: marker size氣泡大小, c: marker color與cmap氣泡顏色)、氣泡邊線顏色與線寬(edgecolors與linewidth)
x = df.sepal_length
y = df.sepal_width
z = df.petal_length*100
 
plt.scatter(x, y, s=z, c=x, cmap="Blues", alpha=0.4, edgecolors="grey", linewidth=2)


#### seaborn熱圖(heatmap)
import seaborn as sns

import numpy as np

df2 = pd.DataFrame(np.random.random((10,10)))
print(df2)
sns.heatmap(df2)


# - 相關係數方陣
# - 注意遮罩矩陣
corr_matrix=df2.corr()
print(corr_matrix)
print(type(corr_matrix))
mask = np.zeros_like(corr_matrix)
mask[np.triu_indices_from(mask)] = True
sns.axes_style("white")
sns.heatmap(corr_matrix, mask=mask, square=True, annot=True)


#### seaborn二維密度估計等值線圖
sns.set_style("white")
# - kde: kernel density estimation
sns.kdeplot(df.sepal_width, df.sepal_length)


# - cmap: color map
sns.kdeplot(df.sepal_width, df.sepal_length, cmap="Blues", shade=True, shade_lowest=True) # try shade_lowest=False


#### pandas DataFrame平行座標圖
# 
# - 多變量繪圖方法
# from pandas.tools.plotting import parallel_coordinates
from pandas.plotting import parallel_coordinates
parallel_coordinates(df, 'species', colormap=plt.get_cmap("Set2")) # input多變量資料框與分組變數


#### 繪圖套件squarify
# 
# - The main function is squarify and it requires two things:
# A coordinate system comprising values for the origin (x and y) and the width/height (dx and dy).
# A list of positive values sorted from largest to smallest and normalized to the total area, i.e., dx * dy).
# (https://github.com/laserson/squarify)
import matplotlib.pyplot as plt
import squarify # conda install -c conda-forge squarify --y

x = pd.DataFrame({
    'size':[98,14,39,53, 61],
    'label':['A', 'B', 'C', 'D', 'E']
})

squarify.plot(sizes=x['size'], label=x['label'], alpha=.4)
plt.axis('off')


#### matplotlib圓餅圖(pie chart)
# 
# - 百分比呈現格式：'%1.1f%%'
plt.pie(x['size'], labels = x['label'], autopct='%1.1f%%')
# plt.axis('equal') # 使比例相等


plt.pie(x['size'], labels = x['label'],
        autopct='%1.1f%%',
        labeldistance=1.2)
# plt.axis('equal')
         
circle=plt.Circle((0,0), 0.5, color='white')
p=plt.gcf()
p.gca().add_artist(circle) # 加上中間白色圓圈


# - wedgeprops引數

# sns.set_style("white")

plt.pie(x['size'], labels = x['label'],
        autopct='%1.1f%%',
        labeldistance=1.2, wedgeprops = { 'linewidth' : 7 })
plt.axis('equal')
         
circle=plt.Circle((0,0), 0.5, color='white')
p=plt.gcf()
p.gca().add_artist(circle)

# 小結：
# https://zwindr.blogspot.com/2017/04/python-matplotlib.html
# 
# Figure
# - 最上層的物件
# - 包含 Axes 物件，甚至其他的元件
# 
# Axes
# - 常用的畫圖物件，包含座標軸，像是直方圖、折線圖、圓餅圖等
# 
# Axis
# - 座標軸物件
# 
# matplotlib.pyplot.gcf()
# - 得到當前的 figure (gcf: get current figure)
# 
# matplotlib.pyplot.gca()
# - 得到當前的 axes (gca: get current axis)

# ![](./matplotlib_parameters.png)


#### **文字雲 wordcloud**
# 
# ```bash
# user~:$conda install -c conda-forge wordcloud --y
# ```
# 
# ```bash
# user~:$conda install -c conda-forge jieba --y
# ```

from wordcloud import WordCloud, ImageColorGenerator
import jieba
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = [15, 10]

# 唐詩三百首
import pandas as pd
df = pd.read_csv('./data/poem.csv')
df.head()

poem = df['詩文'].tolist()

poem = ''.join(poem)

seglist = jieba.cut(poem)

seglist = list(seglist)

seg_df = pd.DataFrame(seglist)[0].value_counts()
seg_df.head()

stop_words = ['，', '。', '？']
text = []
for i in range(len(seglist)):
    w = seglist[i]
    if w not in stop_words:
        text.append(w)

# pd.Series(text).value_counts().head()

text = ' '.join(text)
wc = WordCloud(font_path = './data/msyh.ttf').generate(text)

plt.imshow(wc)
plt.axis("off")

# wordcloud預設不支援中文，可透過Google Noto Fonts https://www.google.com/get/noto/ 下載開源字體

wc = WordCloud(font_path='./data/NotoSansMonoCJKtc-Regular.otf',
               background_color="white",
               max_words = 500)
wc = wc.generate(text)

plt.imshow(wc)
plt.axis("off")

mask = np.array(Image.open("./data/parrot.png"))

wc = WordCloud(font_path='./data/NotoSansMonoCJKtc-Regular.otf',
               background_color="white",
               max_words = 200,
               mask = mask,
               contour_width=3,
               contour_color='steelblue')
wc = wc.generate(text)

plt.imshow(wc)
plt.axis("off")

image_colors = ImageColorGenerator(mask)
plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")

#### **圖形文法繪圖**

#### 快速入門
# plotnine之ggplot通用語法如下： ####

# ggplot(data, mapping=aes(x=var1, y=var2)) + geom_plotname1() + geom_plotname2() + …… + stat_method1() + …… + additional_settings1 + …… 
# 底部圖層(定義資料集、座標及變數的對應) + 幾何繪圖物件1(通常是統計圖形名稱) + 幾何繪圖物件2 + …… + 統計提煉方法1(通常是統計計算方法名稱) + …… + 視覺美化額外設定1 + ……

#### Sample 1: meat資料集
# 套件載入
# import ggplot as gp

# if there is an error related to the pandas
# https://stackoverflow.com/questions/58143253/module-pandas-has-no-attribute-tslib
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
# dir(gp)

import plotnine as gp # !conda install conda-forge::plotnine --y
from plotnine.data import meat

# 查看meat資料
print(type(meat))
meat.head(10)

meat.info()
dir(gp)
# 產生一個以 meat 表格中 date 作為 X 軸 beef 作為 Y 軸的點圖
gp.ggplot(meat, gp.aes(x='date',y='beef')) + gp.geom_point(color='red') + gp.ggtitle('Scatter diagram') + gp.theme(axis_text_x=gp.element_text(angle=30)) # axis.text.x in R ggplot2, but x_axis_text in Python ggplot, and axis_text_x in Python plotnine, ??plot_margin = dict(bottom=0.2)

#### Sample 2: diamonds資料集
# 套件載入
# import ggplot as gp
# import plotnine as gp1 # !conda install conda-forge::plotnine --y
import pandas

# 鑽石資料集資料理解
# 
# - `diamonds` 資料集中包含了大約五萬多顆鑽石的資料，其中包含鑽石的 4C 品質指標，亦即顏色（color）、通透度（clarity）、切割等級（Cut）與重量克拉數（carat），另外還包含了幾個鑽石的尺寸資訊，詳細說明可參考 `diamonds` 的線上手冊。
from plotnine.data import diamonds

print(diamonds.head(10))

print(diamonds.info())

print(diamonds.describe(include='all'))

# 隨機抽取1000筆資料進行繪圖
# 
# - Reference: https://stackoverflow.com/questions/15923826/random-row-selection-in-pandas-dataframe

# Randomly sample 1000 elements from your dataframe
dsmall = diamonds.sample(n=1000)
dsmall

#### plotnine之ggplot: 散佈圖 - 1
# 
# - 底部圖層為兩量化變數price與carat分別對應到雙軸，因子變數color對應至顏色
# - 點圖層是尺寸大小為100的點繪圖
# - Reference : http://ggplot.yhathq.com/docs/geom_point.html
dsmall.color.value_counts() # 七種顏色
gp.ggplot(dsmall, gp.aes(x='carat', y='price', color='color')) + gp.geom_point()


#### plotnine之ggplot: 散佈圖 - 2
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
#### ggplot無法繪出迴歸直線，顯然實現有問題！
gp.ggplot(dsmall, gp.aes('carat', 'price')) + gp.geom_point() + gp.stat_smooth(method="lm", se=False, color="blue")


#### plotnine之ggplot: 散佈圖 - 3
# 
# - 加上信賴區間

gp.ggplot(dsmall, gp.aes('carat', 'price')) + gp.geom_point() + gp.stat_smooth(method="lm", se=True, color="blue")

# get_ipython().run_line_magic('pinfo', 'gp.stat_smooth')

#### plotnine之ggplot: 散佈圖 - 4
# 
# - 底圖層以`colour = 'color'`將資料分組 (其實是多餘的！)
# - 點圖層再依color給各組不同顏色，點尺寸為5
# - 線圖層也依不同顏色繪製配適直線
# - Reference: https://mesfind.github.io/python-ecology-lesson/06-visualization-ggplot-python/

# gp.ggplot(dsmall, gp.aes('carat', 'price', colour='color')) + gp.geom_point(colour='color', size=5) + gp.stat_smooth(colour='color', method="lm", se=False) # stat_smooth圖層不能設colour!

#   File "/opt/anaconda3/lib/python3.7/site-packages/matplotlib/colors.py", line 233, in _to_rgba_no_colorcycle
#     raise ValueError("Invalid RGBA argument: {!r}".format(orig_c))

# ValueError: Invalid RGBA argument: 'color'

#### plotnine之ggplot: 散佈圖 - 5
# 
# - 最簡潔的寫法應該是在底圖層定義`color='color'`(`colour='color'`)，而非底圖層之`group = color`，及/或點線圖層加上`color = color` !
# - 重要觀念！前面圖層的設定，會影響到後面的圖層
# - 較簡潔的寫法如下：

gp.ggplot(dsmall, gp.aes('carat', 'price', colour='color')) + gp.geom_point(size=5) +  gp.stat_smooth(method="lm", se=False)


# 也可以這樣寫
p = gp.ggplot(dsmall, gp.aes('carat', 'price', colour = 'cut'))
p = p + gp.geom_point()
p

#### Sample 3: iris資料集
#### plotnine之ggplot: 線圖
# 
# - 三種鳶尾花共150株的iris資料集
# - 共四種特徵(sepal length, sepal width, petal length, petal width)
# - 有三種品種的鳶尾花(setosa, versicolor, virginica)
# 

from sklearn import datasets # conda install -n viz sklearn
import pandas as pd
iris = datasets.load_iris()

print(iris.data)
print(type(iris.data))
print(iris.data.shape)
print(iris.target_names) 
print(iris.target.shape)
print(iris.feature_names) 
print(iris.target)

import numpy as np
X = pd.DataFrame(iris.data[:, :4],columns=['sepal length','sepal width','petal length','petal width'])
arrsp=iris.target
arrsp=np.where(arrsp==0, 'setosa', arrsp) # 替換 0 為花種字串
arrsp=np.where(arrsp=='1', 'versicolor', arrsp) # 注意！1和2已經變成字串型別了
arrsp=np.where(arrsp=='2', 'virginica', arrsp)

X['Species'] = arrsp
X

gp.ggplot(X, gp.aes(x='petal length', y='petal width', color='Species')) + gp.geom_line()


#### Case 2: 保險客戶資料探索

# from ggplot import *
import plotnine as gp

import pandas as pd

custdata = pd.read_csv('./data/custdata.tsv',sep='\t')

custdata.dtypes

custdata.head()

custdata.describe(include='all')

import missingno as msno # !conda install -c conda-forge missingno --y
import plotnine as gp 

msno.matrix(custdata)

custdata['is.employed'].value_counts(dropna=False)

custdata['housing.type'].value_counts(dropna=False)

custdata['recent.move'].value_counts(dropna=False)

custdata['num.vehicles'].value_counts(dropna=False)

# is.employed近1/3為遺缺值，Why?  
# housing.type, recent.move, num.vehicles均遺缺56個值，NAs個數都一樣，且數量不會太多?

msno.bar(custdata)

# 年齡為0或超過110者為離群值，可能是資料輸入錯誤，或有其它意義 ~ 年齡未知或拒絕回答，有些可能真的是長壽的顧客。

custdata.age.describe()

# 收入負值可能代表不良的資料，也可能有特殊意義。  
# 例如：負債。還是得檢查這個問題有多嚴重，並決定將之去除，或是把負值轉為0。

custdata.income.describe()

# 變數衡量單位的影響：變數Income ***(注意！首字大寫)*** 的定義為"Income custdata['income']/1000". 如果你事先未知此定義，可能會將之解讀為時薪，或年薪(in $1,000)
# ***So, please*** Check資料字典與說明文件
Income = custdata['income']/1000
Income.describe()

# Nullity correlation
msno.heatmap(custdata)

# Nullity correlation ranges from -1 (if one variable appears the other definitely does not) to 0 (variables appearing or not appearing have no effect on one another) to 1 (if one variable appears the other definitely also does).

corr_mat = custdata[['is.employed', 'housing.type', 'recent.move', 'num.vehicles']].isnull().corr()

import numpy as np

# https://github.com/ResidentMario/missingno
# df is a pandas.DataFrame instance
# custdata_nullity = custdata.iloc[:, [i for i, n in enumerate(np.var(custdata.isnull(), axis='rows')) if n > 0]]
# corr_mat = custdata_nullity.isnull().corr()

msno.dendrogram(custdata)
# To interpret this graph, read it from a top-down perspective. Cluster leaves which linked together at a distance of zero fully predict one another's presence—one variable might always be empty when another is filled, or they might always both be filled or both empty, and so on. In this specific example the dendrogram glues together the variables which are required and therefore present in every record.

# - 利用圖形與視覺化找出問題
#   - 對單一變數以視覺化的方式查核其分佈假設
#   - 組寬(binwidth)參數設為五年 (預設值為資料範圍/30).
#   - fill設定直方圖中長條的顏色 (預設值: black).
gp.ggplot(gp.aes(x = 'age'), custdata) + gp.geom_histogram(binwidth=5, fill="gray")

# 搭配密度圖說故事 (What stories are inside/behind?)

gp.ggplot(gp.aes(x='age'), custdata) + gp.geom_density()

# 對單一變數以視覺化的密度曲線圖查核其分佈假設

gp.ggplot(gp.aes(x='income'), custdata) + gp.geom_density()

# 婚姻狀況(類別變數)分佈

gp.ggplot(gp.aes(x='marital.stat'), custdata) + gp.geom_bar(fill="gray")

# 居住地區(類別變數)分佈
# import plotnine as gp1 # 以下使用plotnine套件
p = gp.ggplot(gp.aes(x='state.of.res'), custdata) + gp.geom_bar(fill="gray") + gp.theme(axis_text_x=gp.element_text(angle=90)) # 橫軸州名轉90度，plotnine的實現較佳！
p

# 只考慮合理年齡與收入值的資料子集  
# 檢視年齡與收入的相關性(correlation)

custdata2 = custdata.copy()
custdata2 = custdata2[(custdata2.age > 0) & (custdata2.age < 100) & (custdata2.income > 0)]

custdata2[['age', 'income']].corr()

# 視覺化的洞見較單一的相關係數更能看出發生了什麼事  
# 收入在55歲以前似乎是漸增的，其後逐漸下降
# https://plotnine.readthedocs.io/en/stable/generated/plotnine.stats.stat_smooth.html
gp.ggplot(gp.aes(x='age', y='income'), custdata2) + gp.geom_point() + gp.stat_smooth(method='lm', colour='blue') + gp.ylim(0, 200000)

# 加入線性迴歸線，但它似乎無法真實地捕捉資料的形狀

gp.ggplot(gp.aes(x='age', y='income'), custdata2) + gp.geom_point() + gp.stat_smooth(colour='blue') + gp.ylim(0, 200000) # method預設為'auto'(Use 'loess' if (n<1000), 'glm' otherwise)，局域平滑迴歸線似乎沒有計算信賴區間

# 平滑曲線更容易看出40歲左右收入遞增，而後在55或60左右傾向遞減  
# 寬帶表示以平滑估計值為中心的標準誤，當資料稀疏時寬帶較寬，資料密集時寬帶較窄

gp.ggplot(gp.aes(x='age', y='income'), custdata2) + gp.geom_point() + gp.stat_smooth(method='lowess', colour='blue') + gp.ylim(0, 200000) # method='lowess'與'loess'配適的結果相同

# 兩個類別變數的堆疊長條圖，易比較各組絕對人數  
# 多數客戶已結婚；鰥寡客戶少，但鰥寡者多有保險

gp.ggplot(gp.aes(x = 'marital.stat', fill = 'health.ins'), custdata) + gp.geom_bar() # position = "stack"

# 並排長條圖容易跨組比較投保或未投保比例，但難以比較每組客戶的絕對人數  

gp.ggplot(gp.aes(x='marital.stat', fill='health.ins'), custdata) + gp.geom_bar(position = "dodge")

# - 堆疊與並排長條圖均不易比較各組的投保與未投保之比值  
# - 填滿式長條圖將各組人數正規化為1，可呈現出各組投保與未投保的相對比例  
# - 投保與為投保比值從大到小為：Widowed -> Married -> Divorced/Separated -> Never Married  

gp.ggplot(gp.aes(x = 'marital.stat', y = '1.05', fill = 'health.ins'), custdata)\
    + gp.geom_bar(stat = "identity", position="fill")\
    + gp.geom_point(size = 0.75, alpha = 0.3)

# - 租屋的未婚最多，房貸中的已婚的最多，房貸償清者也是已婚的最多

custdata2['housing_type'] = pd.Categorical(custdata2['housing.type'])
custdata2['housing_type'] = custdata2['housing_type'].cat.codes

custdata3 = custdata2.copy()
custdata3 = custdata3.dropna(how='any')

gp.ggplot(gp.aes(x='housing_type', fill='marital.stat'), custdata3)\
    + gp.geom_bar()\
    + gp.theme(axis_text_x = gp.element_text(angle = 45, hjust = 1))

# 有時若一變數有眾多類別時，則圖形會很擁擠！建議使用分面(faceting)的方式呈現

gp.ggplot(gp.aes(x='housing_type', fill='marital.stat'), custdata3)\
    + gp.geom_bar(position='dodge')\
    + gp.facet_wrap('housing_type', scales="free_y")\
    + gp.theme(axis_text_x = gp.element_text(angle = 45, hjust = 1))

#### Supplement: Mosaic Plot in Python
#### https://sukhbinder.wordpress.com/2018/09/18/mosaic-plot-in-python/
# A mosaic plot allows visualizing multivariate categorical data in a rigorous and informative way.

from statsmodels.graphics.mosaicplot import mosaic
import matplotlib.pyplot as plt
import pandas
 
gender = ['male', 'male', 'male', 'female', 'female', 'female']
pet = ['cat', 'dog', 'dog', 'cat', 'dog', 'cat']
data = pandas.DataFrame({'gender': gender, 'pet': pet})
mosaic(data, ['pet', 'gender'])
plt.show()

import seaborn as sns # conda install -n viz seaborn
 
tips = sns.load_dataset('tips')
mosaic(tips, ['sex','smoker','time'])
plt.show()

#### **資料品質評估 & 圖形文法繪圖**

# 載入ggplot前，需先修正.../site-packages/ggplot/stats/smoothers.py
# ```python
# from pandas.lib import Timestamp
# ```
# 
# 改為
# ```python
# from pandas import Timestamp
# ```
#   
#   
# .../site-packages/ggplot/ggplot.py  602行
# 
# ```python
# fill_levels = self.data[[fillcol_raw, fillcol]].sort(fillcol_raw)[fillcol].unique()
# ```
# 改為
# 
# ```python
# fill_levels = self.data[[fillcol_raw, fillcol]].sort_values(fillcol_raw)[fillcol].unique()
# ```
#     
# 
# 以及.../site-packages/ggplot/stats/smoothers.py
# 
# ```python
# smoothed_data = smoothed_data.sort('x')
# ```
# 
# 改為  
# 
# ```python
# smoothed_data = smoothed_data.sort_values('x')
# ```


#### **補充：互動式繪圖**
#### Bokeh

## 單一折線圖
# Jupyter Notebook on Mac OSX
from bokeh.plotting import figure, output_file, show # !conda install bokeh::bokeh --y
from bokeh.io import push_notebook, show, output_notebook

x = [1, 2, 3, 4, 5]
y = [6, 7, 2, 4, 5]
z = [2, 5, 7, 3, 1]

# 指定圖片輸出的html檔名
output_file("lines.html")
# 若要在jupyter中顯示圖形，使用output_notebook()
# output_notebook()

# 建立新圖形並設定標題與xy軸標籤
p = figure(title="Simple Line Example", x_axis_label='x', y_axis_label='y')

# 加入數列、圖例、線寬
p.line(x, y, legend_label="line", line_width=5)

p.circle(x, z, size = 12, color = 'red')

# 顯示圖片
show(p) # 產生並在瀏覽器中顯示lines.html檔案

import pandas as pd
import numpy as np
date = pd.date_range('2018-08-01', '2018-08-31')
ts = pd.DataFrame({'Date' : date , 
                   'Price': np.random.randint(low = 0, 
                                              high = 100, 
                                              size=len(date)) })

p = figure(width=800, height=250, x_axis_type="datetime") # plot_width=800, plot_height=250
p.line(ts.Date, ts.Price, color='navy', alpha=0.5)
show(p)

#### 內政部不動產成交案件
df = pd.read_csv('./data/lvr_landAcsv/A_LVR_LAND_A.CSV', index_col=False, skiprows=[1]) # Try skiprows=1
df.columns
df.index

# df.head().T

# df = df[1:].reset_index(drop=True)

df.head()

df.dtypes
df['交易年月日'] = df['交易年月日'].astype(int)+19110000
df['交易年月日'] = pd.to_datetime(df['交易年月日'], format = '%Y%m%d')

df['yearmon'] = df['交易年月日'].dt.strftime('%Y%m') 

from bokeh.models.ranges import Range1d
from bokeh.models import LinearAxis

df['單價(元/平方公尺)'] = df['單價(元/平方公尺)'].astype(float)
mean_price = df[['單價(元/平方公尺)', 'yearmon']].groupby(['yearmon']).agg({'單價(元/平方公尺)':'mean'})
trade_num = df.yearmon.value_counts()

x = mean_price.index.tolist()
y = mean_price['單價(元/平方公尺)']
y2 = trade_num 

p = figure(x_range=x, 
           height = 400,
           width = 1000,
           title="土地移轉單價與筆數")

p.extra_y_ranges = {"foo": Range1d(start=0, end=trade_num.max())} # 顯示各月交易筆數的另一軸
p.add_layout(LinearAxis(y_range_name="foo"), 'right')

p.vbar(x=x, top=y, width=0.9) # 單價長條圖
p.line(x=x, y=y2, y_range_name="foo", color='red') # 筆數折線圖
p.xgrid.grid_line_color = None
p.y_range.start = 0
show(p)

df['build_type'] = df['建物型態'].str.replace('\(.*\)', '', regex=True) # regular expression: (#@%$....)

type_count = df.build_type.value_counts().reset_index() # 0 ~ 8 & index & build_type & count

type_count.columns = ['Type', 'Num'] # 'index' & 'build_type' & 'count' -> 'Type' & 'Num'
type_count

from bokeh.palettes import Category20c
from math import pi
from bokeh.transform import cumsum

type_count['angle'] = type_count.Num / sum(type_count.Num) * 2 * pi # 計算角度，一圈360度
type_count['color'] = Category20c[len(type_count)] # 設定顏色

p = figure(height=350, title="Pie Chart", toolbar_location=None,
           tools="hover", tooltips="@Type: @Num")

p.wedge(x=0, y=1, radius=0.4,
        start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
        line_color="white", fill_color='color', legend_label='Build Type', source=type_count) # source: data source; start_angle: include_zero

p.axis.axis_label=None
p.axis.visible=False
p.grid.grid_line_color = None

show(p)

#### plotly
import chart_studio.plotly as py # chart-studio chart_studio
# ImportError: 
# The plotly.plotly module is deprecated,
# please install the chart-studio package and use the
# chart_studio.plotly module instead.
import plotly.graph_objs as go
import pandas as pd

# Jupyter Notebook
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

# Spyder (Plotly doesn't show charts on Spyder (Python 3.5) https://stackoverflow.com/questions/49944041/plotly-doesnt-show-charts-on-spyder-python-3-5)
from plotly.offline import plot


df = pd.read_csv('./data/2330.csv')
df.head()

df['date'] = pd.DataFrame(df.date)
df = df.set_index('date')

trace = go.Candlestick(x=df.index,
                       open=df.open,
                       high=df.high,
                       low=df.low,
                       close=df.close)
data = [trace]
# iplot(data)

# The second graph below is called a rangeslider (See https://plot.ly/python/range-slider/ 32) and you can click and drag the handles (white rectangles) to zoom and pan the figure above.
# https://community.plotly.com/t/go-candlestick-showing-a-second-smaller-graph-not-requested/13342

plot(data, filename='candlestick.html')

#### **補充：歷年性別人口數**
# https://www.ris.gov.tw/346
# 歷年全國人口統計資料（括弧內為資料起始年）
# A 戶數、人口數及遷入、遷出
# 05年底人口按性別及年齡(35

# require xlrd
import pandas as pd
import altair as alt # !conda install -c conda-forge altair-all --y or !conda install anaconda::altair --y
from altair.expr import datum, if_
from vega_datasets import data
# conda install -c conda-forge altair vega vega_datasets

# alt.renderers.enable('notebook')

df = pd.read_excel('./data/y1s600000.xls', header=2, sheet_name='5歲年齡')

df.head()

df = df[df['性別'] != '計'] # 移除總計(218 -> 146)

df.columns

df = df.drop(['年　　別','總　　計','Unnamed: 25'], axis=1)

columns = ['year', 'sex'] # 頭兩欄的欄名

columns.extend([(i+4) for i in range(0,105,5)]) # 往後延伸年齡區間的終止年欄名

df.columns = columns

df.tail()

df = df[:-2] # 前包後不包

df = df.assign(year = df.year.fillna(method='ffill')) # 向前填補遺缺年份

df.isnull().sum()

df = df.fillna(0) # 0值填補剩餘遺缺值

# Wide to long
df = df.melt(['year', 'sex'], var_name='age', value_name='people')

df.dtypes

dat = df.to_dict('records')

df.year = df.year.astype(float)
df.age = df.age.astype(int)
df.people = df.people.astype(int)

slider = alt.binding_range(min = df.year.min(), max = df.year.max(), step=1)
select_year = alt.selection_single(name='year', fields=['year'], bind=slider)

base = alt.Chart(df).add_selection(
    select_year
).transform_filter(
    select_year
).transform_calculate(
    gender=if_(datum.sex == '男', '男', '女')
)

title = alt.Axis(title='人口數')
color_scale = alt.Scale(domain=['男', '女'],
                        range=['#1f77b4', '#e377c2'])

left = base.transform_filter(
    datum.gender == '男'
).encode(
    y=alt.X('age:O', axis=None, sort=alt.SortOrder('descending')),
    x=alt.X('sum(people):Q', axis=title, sort=alt.SortOrder('descending'))
#     ,color=alt.Color('sex:N', scale=color_scale, legend=None)
).mark_bar(color='#1f77b4').properties(title='男')

middle = base.encode(
    y=alt.X('age:O', axis=None, sort=alt.SortOrder('descending')),
    text=alt.Text('age:Q'),
).mark_text().properties(width=20)

right = base.transform_filter(
    datum.gender == '女'
).encode(
    y=alt.X('age:O', axis=None, sort=alt.SortOrder('descending')),
    x=alt.X('sum(people):Q', axis=title)
#     ,color=alt.Color('sex:N', scale=color_scale, legend=None)
).mark_bar(color='#e377c2').properties(title='女')

left | middle | right


#### **補充：地理資訊視覺化**
#### folium
import folium # !conda install conda-forge::folium --y

map_osm = folium.Map(location=(25.1, 121.5))
map_osm


# 地圖樣式類型設定
# - OpenStreetMap
# - Stamen Terrain
# - Stamen Toner
# - Stamen Watercolor
# - CartoDB positron
# - CartoDB dark_matter
# - Mapbox Bright
# - Mapbox Control Room



folium.Map(location=(25.1, 121.5), tiles='CartoDB positron', zoom_start=8)


# 加入標記點

map_1 = folium.Map(location=[25.1, 121.5],
                   zoom_start=11,
                   tiles='OpenStreetMap')
folium.Marker((25.033, 121.564099), popup='Taipei 101').add_to(map_1) # 點擊後出現說明文字
folium.CircleMarker((25.033, 121.564099)).add_to(map_1)
map_1




# get json links
from urllib.request import Request, urlopen, quote
import json
import pandas as pd

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


link = 'http://www.geologycloud.tw/data/zh-tw/liquefaction?area={}&classify={}&all=true'
city = [quote('臺北')] * 3
trend = [quote(t) for t in ['低潛勢', '中潛勢', '高潛勢']]
links = [link.format(x, y) for x, y in zip(city, trend)]
links




def get_data(url):
    u = urlopen(url)
    dat = json.dumps(json.loads(u.read().decode('utf-8')))
    return dat

low = get_data(links[0])
median = get_data(links[1])
high = get_data(links[2])

import folium
map = folium.Map(location=[25.05, 121.5], zoom_start=12, tiles='Stamen Terrain')

folium.Choropleth(
    geo_data=low,
    fill_opacity=0.3,
    line_weight=2,
    fill_color='green'
).add_to(map)

folium.Choropleth(
    geo_data=median,
    fill_opacity=0.3,
    line_weight=2,
    fill_color='yellow'
).add_to(map)

folium.Choropleth(
    geo_data=high,
    fill_opacity=0.3,
    line_weight=2,
    fill_color='red'
).add_to(map)

map

# map.choropleth(low, fill_color='green')
# map.choropleth(median, fill_color='yellow')
# map.choropleth(high, fill_color='red')
# map

# Try:all all data on map

# pip install pyshp or conda install conda-forge::pyshp
import shapefile

#### 空氣品質監測站案例
# reader = shapefile.Reader('./data/空氣品質監測站/空氣品質監測站位置圖_121_10704.shp')
# reader.fields

# fields = reader.fields[1:]
# field_names = [field[0] for field in fields]
# field_names

# buffer = []
# for sr in reader.shapeRecords():
#     records=sr.record
#     for x in [0,2,3,4,5,8]:
#         records[x] = records[x].decode('big5')
#     atr = dict(zip(field_names, records))
#     geom = sr.shape.__geo_interface__
#     geom['coordinates'] = tuple([float(records[6]), float(records[7])])
#     # geom = sr.shape.__geo_interface__ ## (189901.75652849016, 2646862.0741260033)
#     buffer.append(dict(type='Feature', geometry=geom, properties = atr))
# # UnicodeDecodeError: 'utf-8' codec can't decode byte 0xa4 in position 0: invalid start byte

# from json import dumps
# geojson = open("空氣品質監測站_10704.json", "w")
# geojson.write(dumps({"type": "FeatureCollection", "features": buffer}, indent=2) + "\n")
# geojson.close()

# import folium
# map = folium.Map(location=[24.05, 121.5], zoom_start=7,tiles='Stamen Watercolor')

# station_info = json.load(open('空氣品質監測站_10704.json'))
# stations=json.dumps(station_info)

# folium.GeoJson(stations).add_to(map)

# map

# import geopandas as gpd

# geo_df = gpd.read_file('./data/空氣品質監測站_10704.json')
# geo_df.head()

# geo_df.plot()

# map = folium.Map(location=[24.05, 121.5], zoom_start=7,tiles='Stamen Watercolor')
# station_info = json.load(open('空氣品質監測站_10704.json'))
# stations=json.dumps(station_info)

# style = lambda x :{'fillColor':'orange'}

# lon = geo_df.TWD97Lon.tolist()
# lat = geo_df.TWD97Lat.tolist()

# site_name = geo_df.SiteName.tolist()
# address = geo_df.SiteAddres.tolist()
# site_type = geo_df.SiteType.tolist()

# for i in range(len(geo_df)):
    
#     popup = '站名 : {}<br>地址 : {}<br>測站類型 : {}'
#     popup = popup.format(site_name[i], address[i], site_type[i])
    
    
#     map.add_child(
#         folium.CircleMarker(
#             location=[lat[i], lon[i]],
#             color='green', 
#             radius=10, 
#             popup=popup,
#             fill=True,
#             fill_opacity=0.5
#         ))

# map

# from folium.plugins import MarkerCluster

# map = folium.Map(location=[24.05, 121.5], zoom_start=7,tiles='Stamen Watercolor')

# feature_group = folium.FeatureGroup()
# marker_cluster = MarkerCluster()


# for i in range(len(geo_df)):
    
#     popup = '站名 : {}<br>地址 : {}<br>測站類型 : {}'
#     popup = popup.format(site_name[i], address[i], site_type[i])
    
#     circel_marker = folium.CircleMarker(location=[lat[i], lon[i]],
#                                         color='green', 
#                                         radius=10, 
#                                         popup=popup,
#                                         fill=True,
#                                         fill_opacity=0.5
#         )
#     marker_cluster.add_child(circel_marker) 
    
# feature_group.add_child(marker_cluster)

# map.add_child(feature_group)
# map

# map2 = map
# points = [[23.5, -180], [23.5, 180]]

# map2.add_child(folium.PolyLine(locations=points, 
#                                weight=8))
# map2

# map.save('stataionMap.html')

#### **補充：登革熱案例**
from folium.plugins import HeatMap

import pandas as pd
dengue = pd.read_csv("./data/Dengue_Daily.csv")
dengue.head()

location = dengue.copy()[['最小統計區中心點Y', '最小統計區中心點X']].head(1000)

location = location.dropna(how='any')

location = location.values.tolist()

map = folium.Map(location=[24.05, 121.5], zoom_start=7,tiles='Stamen Watercolor')
map.add_child(HeatMap(data=location))

from folium.plugins import HeatMapWithTime

dengue['year'] = dengue['發病日'].str.slice(0, 4)

dengue['year'] = dengue.year.astype(int)

location = dengue.copy()[['最小統計區中心點Y', '最小統計區中心點X']]

l1 = location[dengue.year == 2014].dropna(how='any').dropna(how='any').head(300).values.tolist()
l2 = location[dengue.year == 2015].dropna(how='any').head(300).values.tolist()
l3 = location[dengue.year == 2016].dropna(how='any').head(300).values.tolist()

l = []

l = [l1, l2, l3]

map = folium.Map(location=[24.05, 121.5], zoom_start=7,tiles='Stamen Watercolor')
map.add_child(HeatMapWithTime(l))
map

from folium import plugins

draw = plugins.Draw(export=True)

draw.add_to(map)
map

map.save("mymap.html")

#### 參考資料

# * Milovanovic, I. (2013), Python Data Visualization Cookbook, Packt Publishing.
# * https://itw01.com/2Q6OEHA.html
# * https://stackoverflow.com/questions/50591982/importerror-cannot-import-name-timestamp
# * https://blog.gtwang.org/r/ggplot2-tutorial-basic-concept-and-qplot/
# * https://blog.csdn.net/just_do_it_123/article/details/50933915
# * https://ithelp.ithome.com.tw/articles/10186652
# * https://ggplot2.tidyverse.org/reference/
# * http://kanchengzxdfgcv.blogspot.com/2017/08/python-ggplot.html
