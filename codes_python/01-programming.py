'''
Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the IDS, CICD, and CISD of NTUB (台北商業大學資訊與決策科學研究所暨智能控制與決策研究室教授兼校務永續發展中心主任); at the ME Dept. and CAIDS of MCUT (2020年借調至明志科技大學機械工程系擔任特聘教授兼人工智慧暨資料科學研究中心主任兩年); the CSQ (2019年起任中華民國品質學會大數據品質應用委員會主任委員); the DSBA (2013年創立台灣資料科學與商業應用協會); and the CARS (2012年創立中華R軟體學會)
Notes: This code is provided without warranty.
'''

#### 1.4.1 Python語言原生資料物件操弄(新版Spyder請再增加一個#)
## ------------------------------------------------------------------------
# 中括弧創建Python 串列，千萬別與R 串列混為一談!
x = [1,3,6,8]
print(x)

# type(x) # It's a list

# dir()是查詢物件x下有何可用屬性與方法的重要函數
dir(x)

## ------------------------------------------------------------------------
# 可以混型存放，參見圖1.8 Python 索引編號從0 開始
x[1] = 'peekaboo'
print(x)
# Python 句點語法，引用串列物件append() 方法
# 添加傳入的元素於串列末端
x.append('dwarf') # 一個傳入值
print(x)
# insert() 方法在指定位置(前)塞入元素
x.insert(1, 'Safari') # 兩個傳入值
print(x)

# Python - dir() - how can I differentiate between functions/method and simple attributes? (https://stackoverflow.com/questions/26818007/python-dir-how-can-i-differentiate-between-functions-method-and-simple-att)
[(name,type(getattr(x,name))) for name in dir(x)]

# pop() 方法將指定位置上的元素移除
x.pop(2)
print(x)
# 以in 關鍵字判斷，序列型別物件中是否包含某個元素
print('Safari' in x)
# 串列串接
print([4, 'A_A', '>_<'] + [7, 8, 2, 3])
# 排序
a = [7, 2, 5, 1, 3]
print(sorted(a))
# 透過字串長度升冪(預設) 排序
b = ['saw', 'small', 'He', 'foxes', 'six']
# 串列物件b 為sorted() 函數的位置(positional) 參數值
# key 為sorted 函數的關鍵字(keyword) 參數,len 是關鍵字參數值
# Python 函數的位置參數必須在關鍵字參數前
# 參見1.6.2 節Python 語言物件導向
print(sorted(b, key=len))

## ------------------------------------------------------------------------
# 小括弧創建Python 值組
y = (1, 3, 5, 7)
print(y)
# 可以省略小括弧
y = 1, 3, 5, 7
print(y)
# 值組中還有值組，稱為巢狀值組(或稱嵌套式值組，串列也可以！勿忘Python計算機科學家的語言，所以資料結構彈性較大)
nested_tup = (4, 5, 6), (7, 8)
print(nested_tup)
# 透過tuple 函數可將序列或迭代物件轉為值組
tup = tuple(['foo', [1, 2], True])
print(tup)
# 值組是不可更改的(immutable)
# tup[2] = False
# TypeError: 'tuple' object does not support item assignment
# 但是值組tup 的第二個元素仍為可變的(mutable) 串列
tup[1].append(3) # This's the append method for list, not tuple !
print(tup)
# 解構(unpacking) 值組
tup = (4, 5, 6)
a, b, c = tup
print(c)
# Python 的變數交換方式
x, y = 1, 2
x, y = y, x
print(x)
print(y)

## ------------------------------------------------------------------------
# 大括弧創建字典
d1 = {'a' : 'some value', 'b' : [1, 2, 3, 4]}
print(d1)
# 字典新增元素方式
d1['c'] = 'baby'
d1['dummy'] = 'another value'
print(d1)
## {'a': 'some value', 'b': [1, 2, 3, 4], 'c': 'baby',
## 'dummy': 'another value'}
# 字典取值
print(d1['b'])
# 字典物件get() 方法可以取值，查無該鍵時回傳'There does
# not have this key.'
print(d1.get('b', 'There does not have this key.'))
# 例外狀況發生
print(d1.get('z', 'There does not have this key.'))
# 判斷字典中是否有此鍵
print('b' in d1)
print('z' in d1)
# 字典物件pop() 方法可以刪除元素，例外處理同get() 方法
print(d1.pop('b','There does not have this key.'))
# 鍵為'b' 的字典元素被移除了
print(d1)
# 例外狀況發生
print(d1.pop('z','There does not have this key.'))
# 取得dict 中所有keys，常用！
print(d1.keys())
# 以list() 方法轉為串列物件，注意與上方結果的差異，後不贅述！
print(list(d1.keys()))
# 取得dict 中所有values
print(d1.values())
print(list(d1.values()))
# 取得dict 中所有的元素(items)，各元素以tuple 包著key 及
# value
print(d1.items())
## dict_items([('a', 'some value'), ('c', 'baby'),
## ('dummy', 'another value')])
# 將兩個dict 合併，後面更新前面
x = {'a':1,'b':2} # 註：課本中為a
y = {'b':0,'c':3} # 註：課本中為b
x.update(y) # 註：課本中為a.update(b)
print(x) # 註：課本中為print(a)
# 兩個串列分別表示keys 與values
# 以拉鍊函數zip() 將對應元素捆綁後轉換為dict
tmp = dict(zip(['name','age'], ['Tommy',20]))
print(tmp)

## ------------------------------------------------------------------------
# set() 函數創建Python 集合物件
print(set([2, 2, 2, 1, 3, 3]))
# 同前不計入重復的元素，所以還是1, 2, 3
print({1, 2, 3, 3, 3, 1})
# 集合物件聯集運算union (or)
a = {1, 2, 3, 4, 5}
b = {3, 4, 5, 6, 7, 8}
print(a | b)
# 集合物件交集運算intersection (and)
print(a & b)
# 集合物件差集運算difference
print(a - b)
# print(b - a) # {8, 6, 7}，差集是非對稱運算
# 集合物件對稱差集(或邏輯互斥) 運算symmetric difference(xor)
print(a ^ b)
# 判斷子集issubset() 方法
a_set = {1, 2, 3, 4, 5}
print({1, 2, 3}.issubset(a_set))
# 判斷超集issuperset() 方法
print(a_set.issuperset({1, 2, 3}))
# 判斷兩集合是否相等== 運算子
print({1, 2, 3} == {3, 2, 1})
# 判斷兩值組是否不等!= 運算子
print({1, 2, 3} != {3, 2, 1})

#### 1.4.2 Python語言衍生資料物件取值
## ------------------------------------------------------------------------
# 載入numpy 套件並簡記為np，方便後續引用
import numpy as np
# 呼叫arange() 方法(類似R 語言seq() 函數)，
# 並結合reshape() 方法創建ndarray 物件(4 橫列5 縱行)
data = np.arange(20, dtype='int32').reshape((4, 5)) # Python預設是橫列導向，而R預設是縱行導向
print(data)
# numpy ndarray 類別
print(type(data))

## ------------------------------------------------------------------------
# 屬性矩陣Ｘ與反應變數y切分
# 留意X 取至倒數第一縱行(前包後不包)，以及y 只取最後一行
X, y = data[:, :-1], data[:, -1] # 兩次指派合而為一
print(X)
print(y)
# y_2D = y.reshape((-1,1))
## ------------------------------------------------------------------------
# 一維取單一元素
print(X[2]) # 取第三橫列
# 二維取單一橫列，結果同上
print(X[2,:])
# 一維取值從給定位置至最末端
# 中括弧取值時同R 語言一樣運用冒號(start:end) 運算子
# 冒號(start:end) 後方留空代表取到盡頭
print(X[2:])
# 二維取值，結果同上
print(X[2:,:])
# 倒數的負索引與間距(從倒數第三縱行取到最末行，取值間距為2)
print(X[2:,-3::2]) # start:end:step (X[2:,-3:4:2]或X[2:,-3:3:2])

## ------------------------------------------------------------------------
import pandas as pd
# 第一次使用pandas的read_excel方法需安裝xlrd套件(!conda install xlrd --y)
# skiprows=1 表示從第2 橫列開始讀取資料(請自行更換讀檔路徑)
fb=pd.read_excel('./_data/facebook_checkins_2013-08-24.xls', skiprows=1)

## ------------------------------------------------------------------------
# 確認其為pandas 資料框物件
type(fb) # pandas.core.frame.DataFrame

# type(fb.longitude) # pandas.core.series.Series

# type(fb.longitude.values) # numpy.ndarray

# dir(fb.longitude) # 有'to_list'方法

# type(fb.longitude.to_list()) # list

# help(fb.longitude.to_list)

# 查詢物件fb 的屬性與方法，內容過長返回部分結果
print(dir(fb)[-175:-170])

## ------------------------------------------------------------------------
# 以pandas DataFrame 物件的dtypes 屬性檢視各欄位資料型別
print(fb.dtypes)

## ------------------------------------------------------------------------
# 請與R 比較語法異同及結果差異
print(fb.tail(n=10)) # tail(fb, n = 10)預設看後五筆樣本

#### A. 以下是以中括弧對DataFrame取值
## ------------------------------------------------------------------------
# 二維資料框取出一維序列，無欄位名稱
print(fb['地標名稱'].head())
# pandas 一維結構Series
print(type(fb['地標名稱']))
# 雙中括弧取出的物件仍為二維結構，有欄位名稱
print(fb[['地標名稱']].head())
# pandas 二維結構DataFrame
print(type(fb[['地標名稱']]))

#### B. 以下是以句點語法對DataFrame取值
## ------------------------------------------------------------------------
# 資料框句點語法取值，無欄位名稱
print(fb.類別.head())
# pandas 一維結構Series(句點語法無法取出二維單欄物件！)
print(type(fb.類別))

#### C. 以下是中括弧搭配loc及iloc方法對DataFrame取值
## ------------------------------------------------------------------------
# 資料框loc 方法取值(注意此處冒號運算子為前包後也包!Seeing is believing.)
# loc 方法用中括弧！！！
print(fb.loc[:10, ['地區','累積打卡數']])

## ------------------------------------------------------------------------
# 資料框iloc() 方法取值(注意此處冒號運算子卻又是前包後不包! Seeing is believing.)
print(fb.iloc[:10, [6, 2]]) # iloc中i 之意是丟索引編號(index)而非字串

## ------------------------------------------------------------------------
# 過時用法在此未執行，因為超過過渡期後會產生錯誤訊息
print(fb.ix[:10, ['latitude', 'longitude']])
print(fb.ix[:10, [3, 4]])

#### 1.4.3 Python語言類別變數編碼
## ------------------------------------------------------------------------
import pandas as pd
# 以原生資料結構巢狀串列建構pandas 資料框
df = pd.DataFrame([['green', 'M', 10.1, 'class1'], ['red',
'L', 13.5, 'class2'], ['blue', 'XL', 15.3, 'class1']])
# 設定資料框欄位名稱
df.columns = ['color', 'size', 'price', 'classlabel']
print(df)

## ------------------------------------------------------------------------
# 定義編碼規則字典
size_mapping = {'XL': 3, 'L': 2, 'M': 1}
# 序列map() 方法完成編碼，並更新size 變數
df['size'] = df['size'].map(size_mapping)
print(df)

## ------------------------------------------------------------------------
# 載入類別
from sklearn.preprocessing import LabelEncoder
# 創建(或稱實作) 類別物件class_le
class_le = LabelEncoder()
# 傳入類別變數進行配適與轉換
y = class_le.fit_transform(df['classlabel'])
# 標籤編碼完成(對應整數值預設從0 開始)
print(y)
# y = LabelEncoder().fit_transform(df['classlabel'])
## ------------------------------------------------------------------------
# 逆轉換回原類別值
print(class_le.inverse_transform(y.reshape(-1, 1)))
# 注意下面兩個資料物件內涵相同，但維度不同！前一維，後二維
print(y)
print(y.reshape(-1, 1))

## ------------------------------------------------------------------------
# 取出欲編碼欄位，轉成ndarray(欄位名稱會遺失)
X = df[['color', 'size', 'price']].values
print(X)
# 先進行color 欄位標籤編碼，因為單熱編碼不能有object！(sklearn 0.22.2 以前舊版本的限制，0.22.2 以後(含)新版本無須先進行標籤編碼！)
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
# color 標籤編碼已完成
print(X)

## ------------------------------------------------------------------------
# 載入單熱編碼類別
from sklearn.preprocessing import OneHotEncoder
# 宣告類別物件ohe
#* ohe = OneHotEncoder(categorical_features=[0]) #  sklearn 0.22.2 以前的用法！
ohe = OneHotEncoder()
# 照預設編碼完後轉為常規矩陣
#* print(ohe.fit_transform(X).toarray())
print(ohe.fit_transform(df.iloc[:,[0,3]]).toarray()) # 單獨挑選待編碼欄位
print(np.hstack((ohe.fit_transform(df.iloc[:,[0,3]]).toarray(), df.iloc[:,1:3].values))) # 須將兩numpy ndarray用值組 tuple 組織起來
# 或者可設定sparse 引數為False 傳回常規矩陣
# ohe=OneHotEncoder(categorical_features=[0], sparse=False)
# print(ohe.fit_transform(X))

## ------------------------------------------------------------------------
# get_dummies() 編碼前
print(df[['color', 'size', 'price']])
# pandas DataFrame 的get_dummies() 方法最為方便
print(pd.get_dummies(df[['color', 'size', 'price']])) # 預設drop_first=False是單熱編碼！

print(pd.get_dummies(df[['color', 'size', 'price']], drop_first=True)) # drop_first=True才是虛擬編碼！

#### 1.6 編程範式與物件導向概念
## ------------------------------------------------------------------------
# Python 泛函式編程語法示例
import numpy as np
# 用builtins 模組中的類別type 查核傳入物件的類別
print(type([1,2,3]))
# 呼叫numpy 套件的std() 函數，輸入為串列物件
print(np.std([1,2,3]))

## ------------------------------------------------------------------------
# Python 物件導向編程語法示例
# 以numpy 套件的array() 函數，將原生串列物件轉換為衍生的
# ndarray 物件
a = np.array([1,2,3])
print(a)
print(type(a))
# 句點語法取用ndarray 物件a 的std() 方法
print(a.std())

## ------------------------------------------------------------------------
# Python 的numpy 套件向量化運算示例
print(np.sqrt(a))

## ------------------------------------------------------------------------
# 運用pandas 序列物件Series 之apply() 方法的隱式回圈
import pandas as pd
# 以pandas 套件的Series() 函數，將原生串列物件轉換為衍生的
# Series 物件
a = pd.Series(a)
print(type(a))
# Python pandas 套件的apply() 方法
print(a.apply(lambda x: x+4))

#### 1.6.2 Python語言物件導向
## ------------------------------------------------------------------------
# 線性回歸梯度陡降參數解法
# 定義類別LinearRegressionGD
class LinearRegressionGD(object):
    # 定義物件初始化方法，物件初始化時帶有兩個屬性
    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter
    # 定義物件的方法fit()，此方法會根據傳入的X 與y 計算屬性
    # w_ 和cost_
    def fit(self, X, y):
        # 隨機初始化屬性w_
        self.w_ = np.random.randn(1 + X.shape[1]) # 加1是為了初始化截距項系數
        # 損失函數屬性cost_
        self.cost_ = []
        # 根據物件屬性eta 與n_iter，以及傳入的(訓練資料)X 與y
        # 計算屬性 w_ 和cost_
        for i in range(self.n_iter):
            output = self.lin_comb(X) # 就是預測值y＾hat
            errors = (y - output) # y_i - y＾hat就是殘差
            self.w_[1:] += self.eta * X.T.dot(errors) # Partial L/Partial b1
            self.w_[0] += self.eta * errors.sum() # Partial L/Partial b0
            cost = (errors**2).sum() / 2.0 # 式(1.1)
            self.cost_.append(cost)
        return self
    # 定義fit 方法會用到的lin_comb 線性組合方法
    def lin_comb(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    # 定義物件的方法predict()
    def predict(self, X):
        return self.lin_comb(X)

## ------------------------------------------------------------------------

# import LinearRegGD # 如果將類別定義函數獨立為一個Python命令檔
# dir(LinearRegGD)
        
# 前段程式碼區塊載入類別後，可發現環境中有LinearRegressionGD
# 此行敘述是Python 單列for 回圈寫法，請參考1.8.2 節Python
# 語言資料匯入及匯出的串列推導(list comprehension)
print([name for name in dir() if name in
["LinearRegressionGD"]])
# 模擬五十筆預測變數，使用numpy 常用函數linspace()
X = np.linspace(0, 5, 50) # linspace(start, stop, num)
print(X[:4]) # 前四筆模擬的預測變數
# 模擬五十筆反應變數，利用numpy.random 模組從標準常態分布產生
# 隨機亂數
y = 7.7 * X + 55 + np.random.randn(50) # rand: random; n: Normal/Gaussian
print(y[:4])

## ------------------------------------------------------------------------
# 實作LinearRegressionGD 類物件lr
lr = LinearRegressionGD(n_iter=350)
# 創建後配適前有eta, n_iter, fit(), lin_comb() 與predict()
# print(dir(lr))
# 尚無w_ 與cost_ 屬性
for tmp in ["w_", "cost_"]:
    print(tmp in dir(lr))
# 確認預設迭代次數已變更為350
print(lr.n_iter)
# 傳入單行二維矩陣X 與一維向量y，以梯度陡降法計算系數
lr.fit(X.reshape(-1,1), y)
# 配適完畢後新增加w_ 與cost_ 屬性
for tmp in ["w_", "cost_"]:
    print(tmp in dir(lr))
# 截距與斜率系數
print(lr.w_)
# 最後三代的損失函數值，隨著代數增加而降低
print(lr.cost_[-3:])
# 預測X_new 的y 值
X_new = np.array([2])
print(lr.predict(X_new))
# X 與y 散佈圖及LinearRegressionGD 配適的線性迴歸直線
# Python 繪圖語法參見4.1 節資料視覺化
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X, y)
ax.plot(X, lr.predict(X.reshape(-1,1)), color='red',
linewidth=2)
ax.set_xlabel('x')
ax.set_ylabel('y')
# fig.savefig('./_img/oo_scatterplot.png')

#### 1.8.2 Python語言資料匯入及匯出
## ------------------------------------------------------------------------
data_dir = "./_data/"
# Python 空字串的join() 方法，類似R 語言的paste0() 函數
fname = ''.join([data_dir, "letterdata.csv"])

## ------------------------------------------------------------------------
# mode 引數預設為'r' 讀取模式
f = open(fname)
# 有read() 方法
print(dir(f)[49:54])
# read() 方法讀檔
data = f.read()
# 記得關閉檔案連結
f.close()
# data 為str 類型物件
print(type(data))

## ------------------------------------------------------------------------
# 類別為str 的data 有712669 個字符
print(len(data))
# split() 方法依換行符號"\n" 將data 切成多個樣本的lines
lines = data.split("\n")
# lines 類型為串列
print(type(lines))
# 檢視第一列發現：一橫列一元素，元素內逗號分隔開各欄位名稱
# Python 串列取值冒號運算子，前包後不包
print(lines[0][:35])
# 再次以split() 方法依逗號切出首列中的各欄名稱
header = lines[0].split(',')
print(header[:6])

## ------------------------------------------------------------------------
# 20002 筆
print(len(lines))
# 注意最末空字串
print(lines[20000:])
# 排除首列欄位名稱與末列空字串
lines = lines[1:20001]

## ------------------------------------------------------------------------
# 第一筆觀測值
print(lines[:1])
# 共兩萬筆觀測值
print(len(lines))

## ------------------------------------------------------------------------
import numpy as np
# 宣告numpy 二維字符矩陣(20000, 17)
data = np.chararray((len(lines), len(header)))
print(data.shape)
# 以enumerate() 同時抓取觀測值編號與觀測值
for i, line in enumerate(lines):
    # 串列推導list comprehension，併入data 的第i 列
    data[i, :] = [x for x in line.split(',')]

## ------------------------------------------------------------------------
# 列印變數名稱
# print(header)
# 列印各觀測值
print(data)

## ------------------------------------------------------------------------
# 1.4.2 節pandas 讀檔指令
# fb = pd.read_excel("./_data/facebook_checkins_2013-08-24.xls"
# , skiprows = 1)
import pandas as pd
data_dir = "./_data/"
fname=''.join([data_dir,'/facebook_checkins_2013-08-24.xls'])
# 本節指定工作表名稱與欄位名所在的橫列數
fb = pd.read_excel(fname, sheet_name='總累積', header = 1)
# 讀入後仍為pandas 套件DataFrame 物件
print(type(fb))

## ------------------------------------------------------------------------
# 檢視前五筆數據
print(fb[['地標名稱', '累積打卡數', '地區']].head())
# 縱向變數名稱屬性columns
print(fb.columns[:3])
# 橫向觀測值索引屬性index(從0 到1000 間距1)
print(fb.index)

