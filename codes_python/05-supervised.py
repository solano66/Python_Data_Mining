'''
Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the IDS, CICD, and CISD of NTUB (臺北商業大學資訊與決策科學研究所暨智能控制與決策研究室教授兼校務永續發展中心主任); at the ME Dept. and CAIDS of MCUT (2020年借調至明志科技大學機械工程系擔任特聘教授兼人工智慧暨資料科學研究中心主任兩年); the CSQ (2019年起任中華民國品質學會大數據品質應用委員會主任委員); the DSBA (2013年創立臺灣資料科學與商業應用協會); and the CARS (2012年創立中華R軟體學會)
Notes: This code is provided without warranty.
'''

#### 5.2.1.1 手機簡訊過濾案例
## ------------------------------------------------------------------------
import pandas as pd
# 讀入手機簡訊資料集
sms_raw = pd.read_csv("./_data/sms_spam.csv")
# type：垃圾或正常簡訊，text：簡訊文字內容
print(sms_raw.dtypes)

# type 次數分佈，ham 佔多數，但未過度不平衡
print(sms_raw['type'].value_counts()/len(sms_raw['type']))

# Python 自然語言處理工具集(Natural Language ToolKit)
import nltk # !conda install nltk --y
# 串列推導完成分詞(nltk.download('punkt') at the first time)(list/tuple/dict comprehension, Python很特別的單行迴圈語法)
token_list0 = [nltk.word_tokenize(txt) for txt in
sms_raw['text']] # A nested or embedded list
print(token_list0[3][1:7])

# Same as below
# token_list1 = []
# for doc in range(0,len(token_list0)):
#     tmp = []
#     for word in token_list0[doc]:
#         tmp.append(word.lower())
#     token_list1.append(tmp)

# 串列推導完成轉小寫(Ibiza 變成ibiza, try "TAIWAN".lower())
token_list1 = [[word.lower() for word in doc]
for doc in token_list0] # doc: 各則簡訊，word: 各則簡訊中的各個字詞
print(token_list1[3][1:7])

# 串列推導移除停用詞(nltk.download('stopwords') at the first time)
from nltk.corpus import stopwords
# 179 個英語停用字詞
print(len(stopwords.words('english')))

# 停用字or 已被移除
token_list2 = [[word for word in doc if word not in stopwords.words('english')] for doc in token_list1]

print(token_list2[3][1:7])

# 串列推導移除標點符號
import string
token_list3 = [[word for word in doc if word not in
string.punctuation] for doc in token_list2]
print(token_list3[3][1:7])

# 串列推導移除所有數字(4 不見了)
token_list4 = [[word for word in doc if not word.isdigit()]
for doc in token_list3]
print(token_list4[3][1:7])

# 三層巢狀串列推導移除字符中夾雜數字或標點符號的情形
token_list5 = [[''.join([i for i in word if not i.isdigit()
and i not in string.punctuation]) for word in doc]
for doc in token_list4] # doc: 各則簡訊，word: 各則簡訊中的各個字詞，i: 各個字詞中的各個字元
# similar to paste() in R
# £10,000 變成£
print(token_list5[3][1:7])

# 串列推導移除空元素
token_list6 =[list(filter(None, doc)) for doc in token_list5]
print(token_list6[3][1:7])

# 載入nltk.stem 的WordNet 詞形還原庫(nltk.download('wordnet') nltk.download('omw-1.4') at the first time)
from nltk.stem import WordNetLemmatizer
# 宣告詞形還原器
lemma = WordNetLemmatizer()
# 串列推導完成詞形還原(needs 變成need)
token_list6 = [[lemma.lemmatize(word) for word in doc]
for doc in token_list6]
print(token_list6[3][1:7])

# 串列推導完成各則字詞的串接
# join() 方法將各則簡訊doc 中分開的字符又連接起來
token_list7 = [' '.join(doc) for doc in token_list6]
print(token_list7[:2])

import pandas as pd
# 從feature_extraction 模組載入詞頻計算與DTM 建構類別
from sklearn.feature_extraction.text import CountVectorizer # 計算詞頻，將各則簡訊向量化 Step 1
# 宣告空模 Step 1
vec = CountVectorizer() # binary: bool, default=False. If True, all non zero counts are set to 1.
# 傳入簡訊配適實模並轉換為DTM 稀疏矩陣X
X = vec.fit_transform(token_list7) # Try token_list6 and you will get an error ! Steps 3&4
dir(vec)

# scipy 套件稀疏矩陣類別(csr, compressed sparse rows)
print(type(X))

# 稀疏矩陣儲存詞頻的方式：(橫，縱) 詞頻
print(X[:2]) # 前兩則簡訊的詞頻在稀疏矩陣中的存放方式

# import sys
# import numpy
# numpy.set_printoptions(threshold=sys.maxsize)
# print(X.toarray()[:2]) # 轉成常規矩陣後，方可見前兩則簡訊的完整詞頻向量
dir(X)

# X 轉為常規矩陣(X.toarray())，並組織為pandas 資料框
sms_dtm = pd.DataFrame(X.toarray(),
columns=vec.get_feature_names_out()) # vec.get_feature_names_out()
# sms_dtm.max(axis=0).max() # 不是0-1編碼(Deep Learning)，是詞頻編碼(Bag of Word, BoW)
# 5559 列(則)7484 行(字) 的結構
print(sms_dtm.shape)

# 模型vec 取出DTM 各字詞的get_feature_names() 方法
print(len(vec.get_feature_names())) # 共有7484 個字詞 vec.get_feature_names_out()

print(vec.get_feature_names()[300:305]) # vec.get_feature_names_out()

import numpy as np # 書中遺漏此列程式碼

# 5559 則簡訊中app 此字只有6 則正詞頻，的確稀疏(新版numpy請用下行註解程式碼)
print(np.argwhere(sms_dtm['app'] > 0)) # 列向量
# print(np.argwhere((sms_dtm['app'] > 0).values.reshape((-1,1)))) # 新版numpy需轉成行向量

# DTM 部分內容
print(sms_dtm.iloc[4460:4470, 300:305])
# sms_dtm.max().max() # 15, 原始詞頻dtm，適合配適multinomialNB()

# 訓練與測試集切分(sms_raw, sms_dtm, token_list6)
sms_raw_train = sms_raw.iloc[:4170, :]
sms_raw_test = sms_raw.iloc[4170:, :]
sms_dtm_train = sms_dtm.iloc[:4170, :]
sms_dtm_test = sms_dtm.iloc[4170:, :]
token_list6_train = token_list6[:4170]
token_list6_test = token_list6[4170:]
# 查核各子集類別分佈
print(sms_raw_train['type'].value_counts()/
len(sms_raw_train['type']))

print(sms_raw_test['type'].value_counts()/
len(sms_raw_test['type']))

# WordCloud() 統計詞頻須跨篇組合所有詞項
tokens_train = [token for doc in token_list6_train
for token in doc]
print(len(tokens_train))

# 邏輯值索引結合zip() 綑綁函數，再加判斷句與串列推導
tokens_train_spam = [token for is_spam, doc in
zip(sms_raw_train['type'] == 'spam' , token_list6_train)
if is_spam for token in doc]
# 取出正常簡訊
tokens_train_ham = [token for is_ham, doc in
zip(sms_raw_train['type'] == 'ham' , token_list6_train)
if is_ham for token in doc]
# 逗號接合訓練與spam 和ham 兩子集tokens
str_train = ','.join(tokens_train)
str_train_spam = ','.join(tokens_train_spam)
str_train_ham = ','.join(tokens_train_ham)

# Python 文字雲套件(conda install -c conda-forge wordcloud --y)
from wordcloud import WordCloud
# 宣告文字雲物件(最大字數max_words 預設為200)
wc_train = WordCloud(background_color="white",
prefer_horizontal=0.5)
# 傳入資料統計，並產製文字雲物件
wc_train.generate(str_train) # str_train -> str_train_spam, str_train_ham
# 呼叫matplotlib.pyplot 模組下的imshow() 方法繪圖
import matplotlib.pyplot as plt
plt.imshow(wc_train)
plt.axis("off")
# plt.show()
# plt.savefig('wc_train.png')
# 限於篇幅，str_train_spam 和str_train_ham 文字雲繪製代碼省略

# 載入多項式天真貝氏模型類別
from sklearn.naive_bayes import MultinomialNB # MultinomialNB: 整數值編碼；BinomialNB: 二元值編碼；GaussianNB: 實數值編碼
# 模型定義、配適與預測
clf = MultinomialNB()

clf.fit(sms_dtm_train, sms_raw_train['type'])
train = clf.predict(sms_dtm_train)
print(" 訓練集正確率為{}".format(sum(sms_raw_train['type'] ==
train)/len(train)))

pred = clf.predict(sms_dtm_test)
print(" 測試集正確率為{}".format(sum(sms_raw_test['type'] ==
pred)/len(pred)))
# dir(clf)

# 訓練所用的各類樣本數
print(clf.class_count_)

# 兩類與7612(7484) 個屬性的交叉列表**(絕對頻次表)**
print(clf.feature_count_)

print(clf.feature_count_.shape)

# 已知類別下，各屬性之條件機率(似然率)Pr[x_i|y] 的對數值**(相對頻次表再取對數值)**
print(clf.feature_log_prob_[:, :4])

print(clf.feature_log_prob_.shape)

# 將對數條件機率轉成機率值(補充程式碼)
feature_prob = np.exp(clf.feature_log_prob_)
print(feature_prob.shape)
print(feature_prob[:, :4])
# 驗證兩類之機率值總和為1(補充程式碼)
print(np.apply_along_axis(np.sum, 1, feature_prob)) # [1. 1.]
# 兩類最大字詞機率值(補充程式碼)
print(np.apply_along_axis(np.max, 1, feature_prob)) # [0.00813987 0.01839848]]
# 抓出兩類機率前五高的字詞，與文字雲結果相符(補充程式碼)
print(sms_dtm.columns.values[np.argsort(-feature_prob)[:,:5]])
# ham: ['nt' 'get' 'go' 'ok' 'call']
# spam: ['call' 'free' 'txt' 'mobile' 'text']

import numpy as np
# 載入sklearn 交叉驗證模型選擇的重要函數
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import sem
# 自定義k 摺交叉驗證模型績效計算函數
def evaluate_cross_validation(clf, X, y, K): # 輸入全樣本集
    # 創建k 摺交叉驗證迭代器(iterator)，用於X 與y 的切分
    cv = KFold(n_splits=K, shuffle=True, random_state=0)
    scores = cross_val_score(clf, X, y, cv=cv)
    print("{}摺交叉驗證結果如下：\n{}".format(K, scores))
    tmp = " 平均正確率：{0:.3f}(+/-標準誤{1:.3f})"
    print(tmp.format(np.mean(scores), sem(scores)))

evaluate_cross_validation(clf, sms_dtm, sms_raw['type'], 5)

#### 5.2.2.1 電離層無線電訊號案例
## ------------------------------------------------------------------------
import numpy as np
import pandas as pd
iono = pd.read_csv("./_data/ionosphere.data", header=None)
# 切分屬性矩陣與目標向量
X = iono.iloc[:, :-1] # 縱向前包後不包
y = iono.iloc[:, -1] # 縱向無冒號，故無所謂前包後不包
print(X.shape)

print(y.shape)

# 無名目屬性，適合k 近鄰學習
print(X.dtypes)

# 資料無遺缺，可直接進行k 近鄰學習
print(" 遺缺{}個數值".format(X.isnull().sum().sum()))
# print(" 各變量遺缺概況：")
# print(X.isnull().sum())

# 訓練集與測試集切分
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
random_state=14) # random_state=1234
print(" 訓練集有{}樣本".format(X_train.shape[0]))

print(" 測試集有{}樣本".format(X_test.shape[0]))

print(" 每個樣本有{}屬性".format(X_train.shape[1]))

print(" 資料集類別分佈為：\n{}.".format(y.value_counts()/len(y)))

print(" 訓練集類別分佈為：\n{}."
.format(y_train.value_counts()/len(y_train)))

print(" 測試集類別分佈為：\n{}."
.format(y_test.value_counts()/len(y_test)))

# 載入sklearn 前處理模組的標準化轉換類別
from sklearn.preprocessing import StandardScaler
# 模型定義(未更改預設設定)、配適與轉換(似乎多餘！)
sc = StandardScaler()
# 配適與轉換接續執行函數fit_transform()
X_train_std = sc.fit_transform(X_train)
# 依訓練集擬合的(**標準化**)模型，對測試集做轉換
X_test_std = sc.transform(X_test)
# sc.mean_ from training set
# sc.var_ from training set

# 整個屬性矩陣標準化是為了交叉驗證調參(注意！模型sc 內容會變)
X_std = sc.fit_transform(X)
# sc.mean_ from whole dataset
# sc.var_ from whole dataset

# 建議比較sc_train = StandardScaler(), sc_train.fit_transform(X_train)以及sc = StandardScaler(), sc.fit_transform(X)兩者sc_train.mean_和sc.mean_以及sc_train.var_和sc.var_的差異！！！

# 載入sklearn 近鄰學習模組的k 近鄰分類類別
from sklearn.neighbors import KNeighborsClassifier
# 模型定義(未更改預設設定)、配適與轉換
estimator = KNeighborsClassifier()
estimator.fit(X_train_std, y_train)
# 模型estimator 的get_params() 方法取出模型參數：
# Minkowski 距離之p 為2(歐幾里德距離) 與鄰居數是5
for name in ['metric','n_neighbors','p']:
    print(estimator.get_params()[name])

# 對訓練集進行預測
train_pred = estimator.predict(X_train_std)
# 訓練集前五筆預測值
print(train_pred[:5])

# 訓練集前五筆實際值
print(y_train[:5])

train_acc = np.mean(y_train == train_pred) * 100
print(" 訓練集正確率為{0:.1f}%".format(train_acc))

# 對測試集進行預測
y_pred = estimator.predict(X_test_std)
# 測試集前五筆預測值
print(y_pred[:5])

# 測試集前五筆實際值
print(y_test[:5])

test_acc = np.mean(y_test == y_pred) * 100
print(" 測試集正確率為{0:.1f}%".format(test_acc))
# 以上是單次保留法的結果，重複多次的保留法更好！

# sklearn 套件中模型選擇模組下交叉驗證訓練測試機制之績效計算函數
from sklearn.model_selection import cross_val_score
# 預設為三摺(新版sklearn預設改為五摺)交叉驗證運行一次
scores = cross_val_score(estimator, X_std, y,
scoring='accuracy')
print(scores.shape)

average_accuracy = np.mean(scores) * 100
print(" 五次的平均正確率為{0:.1f}%".format(average_accuracy))

# std_accuracy = np.std(scores, ddof=1) * 100 # 留意np.std()是計算母體標準差
# print(" 五次的正確率標準差為{0:.1f}%".format(std_accuracy))

# 逐步收納結果用
avg_scores = []
all_scores = []
# 定義待調參數候選集
parameter_values = list(range(1, 32, 3)) # range(1, 31)
# 對每一參數候選值，執行下方內縮敘述
for n_neighbors in parameter_values:
    # 宣告模型規格n_neighbors
    estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
    # cross_val_score() 依模型規格與資料集進行交叉驗證訓練和測試
    sc=cross_val_score(estimator,X_std,y,scoring='accuracy')
    # 績效分數(accuracy) 平均值計算與添加
    avg_scores.append(np.mean(sc))
    all_scores.append(sc)
# 近鄰數從1 到20 的平均正確率
print(len(avg_scores))

print(avg_scores)

# 近鄰數從1 到20 的五摺交叉驗證結果
print(len(all_scores))

# 不同近鄰數k 值下，五次交叉驗證的正確率
print(all_scores[:4])

# 不同近鄰數下平均正確率折線圖
from matplotlib import pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
plt.xticks(np.arange(1, 32, 3)) # np.arange(1, 21)
ax.plot(parameter_values, avg_scores, '-o')
# fig.savefig('./_img/iono_tuning_avg_scores.png')

# 載入sklearn 前處理模組正規化轉換類別
from sklearn.preprocessing import MinMaxScaler
# 載入統計機器學習流程化模組
from sklearn.pipeline import Pipeline
# 流程定義
pipe = Pipeline([('scale', MinMaxScaler()), ('predict',
KNeighborsClassifier())])
# 流程與資料傳入cross_val_score() 函數
scores = cross_val_score(pipe, X, y, scoring='accuracy')
# 五摺交叉驗證結果
print(" 五次正確率結果為{}%".format(scores*100))

print(" 平均正確率為{0:.1f}%".format(np.mean(scores) * 100))

print(" 五次正確率的標準差為{0:.1f}%".format(np.std(scores, ddof=1) * 100))

#### 5.2.3.1 光學手寫字元案例
## ------------------------------------------------------------------------
import pandas as pd
letters = pd.read_csv("./_data/letterdata.csv")
# 檢視變數型別
print(letters.dtypes)

# 各整數值變數介於0 到15 之間(4 bits 像素值)
print(letters.describe(include = 'all'))

# 目標變數各類別分佈平均(預設依各類頻次降冪排序)
print(letters['letter'].value_counts()) # np.unique(letters.letter, return_counts=True)

# 載入sklearn 屬性挑選模組的變異數過濾類別
from sklearn.feature_selection import VarianceThreshold
# 模型定義、配適與轉換(i.e. 刪除零變異屬性)
vt = VarianceThreshold(threshold=0)
# 並無發現零變異屬性
print(vt.fit_transform(letters.iloc[:,1:]).shape)

# 沒有超過(低於或等於) 變異數門檻值0 的屬性是0 個
import numpy as np
print(np.sum(vt.get_support() == False)) # Get a mask of the features selected
# vt.get_support(indices=True) # Get integer index of the features selected

# 計算相關係數方陣後轉numpy ndarray
cor = letters.iloc[:,1:].corr().values
print(cor[:5,:5])

# 相關係數超標(+-0.8) 真假值方陣
import numpy as np
np.fill_diagonal(cor, 0) # 變更對角線元素值為0
threTF = abs(cor) > 0.8
print(threTF[:5,:5])

# 類似R 語言的which(真假值矩陣, arr.ind=TRUE)
print(np.argwhere(threTF == True))

# 核對變數名稱，注意相關係數計算時已排除掉第1 個變數letter
print(letters.columns[1:5])

# pandas 資料框boxplot() 方法繪製並排盒鬚圖(26組數據)
ax1 = letters[['xbox', 'letter']].boxplot(by = 'letter')
fig1 = ax1.get_figure()
# fig1.savefig('./_img/xbox_boxplot.png')
ax2 = letters[['ybar', 'letter']].boxplot(by = 'letter')
fig2 = ax2.get_figure()
# fig2.savefig('./_img/ybar_boxplot.png')

# 訓練與測試集切分
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
letters.iloc[:, 1:], letters['letter'], test_size=0.2,
random_state=0)
# 數據標準化
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# 計算X_train 各變數的mu 和sigma
sc.fit(X_train)
# 真正做轉換
X_train_std = sc.transform(X_train)
# 以X_train 各變數的mu 和sigma 對X_test 做轉換
X_test_std = sc.transform(X_test)

# SVC: 支援向量分類(Support Vector Classification)
# SVR: 支援向量迴歸(Support Vector Regression)
# OneClassSVM: 非監督式離群偵測(Outlier Detection)
from sklearn.svm import SVC
# 模型定義(使用線性核函數，先前預設的核函數為'linear'，現已改為'rbf')、配適與轉換
svm = SVC(kernel='poly') # 原來括弧內是空白
svm.fit(X_train_std, y_train)
tr_pred = svm.predict(X_train_std)
y_pred = svm.predict(X_test_std)
# 訓練集前5 筆預測值
print(tr_pred[:5])

# 訓練集前5 筆實際值
print(y_train[:5].tolist())

# 測試集前5 筆預測值
print(y_pred[:5])

# 測試集前5 筆實際值
print(y_test[:5].tolist())

# 注意Python 另一種輸出格式化語法(% 符號)
err_tr = (y_train != tr_pred).sum()/len(y_train)
print(' 訓練集錯誤率為：%.5f' % err_tr)

# 測試集錯誤率稍高於訓練集的錯誤率
err = (y_test != y_pred).sum()/len(y_test)
print(' 測試集錯誤率為：%.5f' % err)

# 變更徑向基底函數之參數gamma 為0.2，(5.34) 式的C 為1.0
svm = SVC(kernel='rbf', random_state=0, gamma=0.2, C=1.0) # You can try SVC().
svm.fit(X_train_std, y_train)
tr_pred = svm.predict(X_train_std)
y_pred = svm.predict(X_test_std)
# 訓練集前5 筆預測值
print(tr_pred[:5])

# 訓練集前5 筆實際值
print(y_train[:5].tolist())

# 測試集前5 筆預測值
print(y_pred[:5])

# 測試集前5 筆實際值
print(y_test[:5].tolist())

err_tr = (y_train.values != tr_pred).sum()/len(y_train)
print(' 訓練集錯誤率為：%.5f' % err_tr)

# 測試集錯誤率也是稍高於訓練集的錯誤率
err = (y_test != y_pred).sum()/len(y_test)
print(' 測試集錯誤率為：%.5f' % err)

# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import confusion_matrix

# 假設 y_test 是您的測試集標籤，y_pred 是模型預測的標籤
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 使用數字範圍作為臨時標籤
num_labels = cm.shape[0]  # 獲取標籤數量
labels = np.arange(num_labels)  # 生成臨時標籤

# 將混淆矩陣轉換為 DataFrame
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

# 如果需要，可以計算所有值的總和
cm_df['all'] = cm_df.sum(axis=1)
cm_df.loc['all'] = cm_df.sum()

# 打印部分結果
print(cm_df.iloc[:12, :12]) 完整可以用這個

############# 以下可以忽略
# 載入整合pandas, scikit-learn 與xgboost 的套件pandas_ml(請注意scikit-learn須降版為0.21.1，pandas須降為0.22.0，最好是另創一個虛擬環境)
# conda install -c conda-forge scikit-learn==0.21.1 pandas==0.22.0 pandas_ml --y
import pandas_ml as pdml
# 注意！須傳入numpy ndarray 物件，以生成正確的混淆矩陣
cm = pdml.ConfusionMatrix(y_test.values, y_pred)
# dir(cm)
# 混淆矩陣轉成pandas 資料框，方便書中結果呈現
cm_df = cm.to_dataframe(normalized=False, calc_sum=True,
sum_label='all')
# 混淆矩陣部分結果
print(cm_df.iloc[:12, :12])

# stats() 方法生成整體(3.2.2.3 節) 與類別相關指標(3.2.2.4 節)
perf_indx = cm.stats()
# 儲存為collections 套件的有序字典結構(OrderedDict)
print(type(perf_indx))

# 有序字典結構的鍵，其中cm 為相同的混淆矩陣
print(perf_indx.keys())

# overall 鍵下也是有序字典結構
print(type(perf_indx['overall']))
# perf_indx['overall'].keys()

# 整體指標內容如下：
print(" 分類模型正確率為：{}".format(perf_indx['overall']
['Accuracy']))

print(" 正確率95% 信賴區間為：\n{}".format(perf_indx
['overall']['95% CI']))

print("Kappa 統計量為：\n{}".format(perf_indx['overall']
['Kappa']))

# class 鍵下是pandas 資料框結構
print(type(perf_indx['class']))

# 26 個字母(縱向) 各有26 個類別(橫向) 相關指標
print(perf_indx['class'].shape)

print(perf_indx['class'])

# 混淆矩陣熱圖視覺化，請讀者自行嘗試
import matplotlib.pyplot as plt
ax = cm.plot()
fig = ax.get_figure()
# fig.savefig('./_img/svc_rbf.png')

#### 5.2.4.1 銀行貸款風險管理案例
## ------------------------------------------------------------------------
import numpy as np
import pandas as pd
# 讀入UCI 授信客戶資料集
credit = pd.read_csv("./_data/germancredit.csv")
print(credit.shape)

# 檢視變數型別
print(credit.dtypes)

# "Default": 是否違約, "checkingstatus1": 支票存款帳戶餘額, "duration": 償款期限(月), "history": 信用記錄, "purpose": 貸款目的, "amount": 貸款金額, "savings": 儲蓄存款帳戶餘額, "employ": 任現職多久, "installment"(the amount of money paid out per unit time, here in percentage of disposable income): 繳費率或賠付率，指單位時間分期付款金額，以可支配所得的比例為單位, "status": 個人婚姻狀況與性別, "others": 有無共同申貸人和保證人, "residence"(present residence since): 居住現址多久, "property": 恆產, "age": 年齡, "otherplans"(other installment plans): 其他分期計劃, "cards"(number of existing credits at this bank): 本行現存信用數, "job": 職業, "liable"(number of people being liable to provide maintenance for): 扶養親屬數, "tele": 名下有無電話, "foreign": 是否為外來工作者, "rent": 是否租屋(derived from housing)

# 目標變數Default(已為0-1 值) 次數分佈
print(credit.Default.value_counts())

# 變數轉換字典target
target = {0: "Not Default", 1: "Default"}
credit.Default = credit.Default.map(target)

# 成批產製類別變數(dtype 為object) 的次數分佈表(存為字典結構)
# 先以邏輯值索引取出object 欄位名稱
col_cat = credit.columns[credit.dtypes == "object"]
# 逐步收納各類別變數次數統計結果用
counts_dict = {}
# 取出各欄類別值統計頻次
for col in col_cat:
    counts_dict[col] = credit[col].value_counts()
# 印出各類別變數次數分佈表
print(counts_dict)

# 代號與易瞭解名稱對照字典
print(dict(zip(credit.checkingstatus1.unique(),["< 0 DM",
"0-200 DM","no account","> 200 DM"])))

# 逐欄轉換易瞭解的類別名稱
credit.checkingstatus1 = credit.checkingstatus1.map(dict(zip
(credit.checkingstatus1.unique(),["< 0 DM","0-200 DM","no account",
"> 200 DM"])))
credit.history = credit.history.map(dict(zip(credit.history.
unique(),["good","good","poor","poor","terrible"])))
credit.purpose = credit.purpose.map(dict(zip(credit.purpose.
unique(),["newcar","usedcar","goods/repair","goods/repair",
"goods/repair","goods/repair","edu","edu","biz","biz"])))
credit.savings = credit.savings.map(dict(zip(credit.savings.
unique(),["< 100 DM","100-500 DM","500-1000 DM","> 1000 DM",
"unknown/no account"])))
credit.employ = credit.employ.map(dict(zip(credit.employ.
unique(),["unemployed","< 1 year","1-4 years","4-7 years",
"> 7 years"])))

# 基於篇幅考量，上面只顯示前五類別變數的轉換代碼，以下請讀者自行執行。
credit.status = credit.status.map(dict(zip(credit.status.unique(),["M/Div/Sep","F/Div/Sep/Mar","M/Single","M/Mar/Wid"])))
credit.others = credit.others.map(dict(zip(credit.others.unique(),["none","co-applicant","guarantor"])))
credit.property = credit.property.map(dict(zip(credit.property.unique(),["none","co_applicant","guarantor"])))
credit.otherplans = credit.otherplans.map(dict(zip(credit.otherplans.unique(),["bank","stores","none"])))

# If you want to determine the house is rent or not 如果要新建一個是否租房的欄位
# credit['rent'] = credit['housing'] == "A151"
# credit['rent'].value_counts()
## del credit['housing']

credit.housing = credit.housing.map(dict(zip(credit.housing.unique(),["own","for free","rent"])))
credit.job = credit.job.map(dict(zip(credit.job.unique(),["unemployed","unskilled","skilled", "mgt/self-employed"])))
credit.tele = credit.tele.map(dict(zip(credit.tele.unique(),["none","yes"])))
credit.foreign = credit.foreign.map(dict(zip(credit.foreign.unique(),["foreign","german"])))

# credit.to_csv("./_data/germancredit_xAxxx.csv")

# 資料表內容較容易瞭解
print(credit.head())

# 除了'property'外均完整
credit.isnull().sum()

# 授信客戶資料摘要統計表
print(credit.describe(include='all'))

# crosstab() 函數建支票存款帳戶狀況，與是否違約的二維列聯表
ck_f = pd.crosstab(credit['checkingstatus1'],
credit['Default'], margins=True)
# 計算相對次數
ck_f.Default = ck_f.Default/ck_f.All
ck_f['Not Default'] = ck_f['Not Default']/ck_f.All
print(ck_f)

# 儲蓄存款帳戶餘額狀況，與是否違約的二維列聯表
sv_f = pd.crosstab(credit['savings'],
credit['Default'], margins=True)
sv_f.Default = sv_f.Default/sv_f.All

sv_f['Not Default'] = sv_f['Not Default']/sv_f.All
print(sv_f)

# 與R 語言summary() 輸出相比，多了樣本數count 與標準差std
print(credit['duration'].describe())

print(credit['amount'].describe())

# 字串轉回0-1 整數值
inv_target = {"Not Default": 0, "Default": 1}
credit.Default = credit.Default.map(inv_target)

# 成批完成類別預測變數標籤編碼 (Python自己來，R自動完成)
from sklearn.preprocessing import LabelEncoder
# 先以邏輯值索引取出類別欄位名稱
col_cat = credit.columns[credit.dtypes == "object"]
# 宣告空模
le = LabelEncoder()
# 逐欄取出類別變數值後進行標籤編碼
for col in col_cat:
    credit[col] = le.fit_transform(credit[col].astype(str))
# credit.property.value_counts()
# 切分類別標籤向量y 與屬性矩陣X
y = credit['Default']
X = credit.drop(['Default'], axis=1)
# 切分訓練集及測試集，random_state 引數設定亂數種子
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.1, random_state=33)

# 訓練集類別標籤次數分佈表
Default_train = pd.DataFrame(y_train.value_counts(sort =
True))
# 計算與建立累積和欄位'cum_sum'
Default_train['cum_sum'] = Default_train['count'].cumsum()
# 計算與建立相對次數欄位'perc'
tot = len(y_train)
Default_train['perc']=100*Default_train['count']/tot
# 計算與建立累積相對次數欄位'cum_perc'
Default_train['cum_perc']=100*Default_train['cum_sum']/tot

### 限於篇幅，書本省略下面程式碼
# 測試集類別標籤次數分佈表
Default_test = pd.DataFrame(y_test.value_counts(sort=True))
# 計算與建立累積和欄位'cum_sum'
Default_test['cum_sum'] = Default_test['count'].cumsum()
# 計算與建立相對次數欄位'perc'
Default_test['perc'] = 100*Default_test['count']/len(y_test)
# 計算與建立累積相對次數欄位'cum_perc'
Default_test['cum_perc'] = 100*Default_test['cum_sum']/len(y_test)
###

# 比較訓練集與測試集類別標籤分佈
print(Default_train)
print(Default_test)

# 載入sklearn 套件的樹狀模型模組tree
from sklearn import tree
# 宣告DecisionTreeClassifier() 類別空模clf(未更改預設設定)
clf = tree.DecisionTreeClassifier()
# 傳入訓練資料擬合實模clf
clf = clf.fit(X_train, y_train)
# ValueError: could not convert string to float: '> 200 DM' (前面如果字串自變數未標籤編碼，則會報錯！)
# 預測訓練集標籤train_pred
train_pred = clf.predict(X_train)
print(' 訓練集錯誤率為{0}.'.format(np.mean(y_train !=
train_pred)))

# 預測測試集標籤test_pred
test_pred = clf.predict(X_test)
# 訓練集錯誤率遠低於測試集，過度配適的癥兆
print(' 測試集錯誤率為{0}.'.format(np.mean(y_test !=
test_pred)))
print('此樹有{}節點'.format(clf.tree_.node_count)) # 好複雜的一棵樹！所以接下來我們調參
print(clf.get_params())
keys = ['max_depth', 'max_leaf_nodes', 'min_samples_leaf']
# type(clf.get_params()) # dict
print([(key, clf.get_params().get(key)) for key in keys])

# 再次宣告空模clf(更改上述三參數設定)、配適與預測
clf = tree.DecisionTreeClassifier(max_leaf_nodes = 10,
min_samples_leaf = 7, max_depth= 30)
clf = clf.fit(X_train,y_train)
train_pred = clf.predict(X_train)
print(' 訓練集錯誤率為{0}.'.format(np.mean(y_train !=
train_pred)))

# 過度配適情況已經改善
test_pred = clf.predict(X_test)
print(' 測試集錯誤率為{0}.'.format(np.mean(y_test !=
test_pred)))

#### 報表生成與樹模繪圖
n_nodes = clf.tree_.node_count
print(' 分類樹有{0} 個節點.'.format(n_nodes))

children_left = clf.tree_.children_left
s1 = ' 各節點的左子節點分別是{0}'
s2 = '\n{1}(-1 表葉子節點沒有子節點)。'
print(''.join([s1, s2]).format(children_left[:9],
children_left[9:]))

children_right = clf.tree_.children_right
s1 = ' 各節點的右子節點分別是{0}'
s2 = '\n{1}(-1 表葉子節點沒有子節點)。'
print(''.join([s1, s2]).format(children_right[:9],
children_right[9:]))

feature = clf.tree_.feature
s1 = ' 各節點分支屬性索引為(-2 表無分支屬性)'
s2 = '\n{0}。'
print(''.join([s1, s2]).format(feature))

threshold = clf.tree_.threshold
s1 = ' 各節點分支屬性門檻值為(-2 表無分支屬性門檻值)'
s2 = '\n{0}\n{1}\n{2}\n{3}。'
print(''.join([s1, s2]).format(threshold[:6],
threshold[6:12], threshold[12:18], threshold[18:]))

# 各節點樹深串列node_depth
node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
# 各節點是否為葉節點的真假值串列
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
# 值組(節點編號, 父節點深度) 形成的堆疊串列, 初始化時只有根節點
stack = [(0, -1)]
# 從堆疊逐一取出資訊產生報表，堆疊最終會變空
while len(stack) > 0:
    node_i, parent_depth = stack.pop()
    # 自己的深度為父節點深度加1
    node_depth[node_i] = parent_depth + 1
    # 如果是測試節點(i.e. 左子節點不等於右子節點)，而非葉節點
    if (children_left[node_i] != children_right[node_i]):
    # 加左分枝節點，分枝節點的父節點深度正是自己的深度
        stack.append((children_left[node_i],parent_depth+1))
    # 加右分枝節點，分枝節點的父節點深度正是自己的深度
        stack.append((children_right[node_i],parent_depth+1))
    else:
    # is_leaves 原預設全為False，最後有True 有False
        is_leaves[node_i] = True

print(" 各節點的深度分別為：{0}".format(node_depth))

print(" 各節點是否為終端節點的真假值分別為：\n{0}\n{1}"
.format(is_leaves[:10], is_leaves[10:]))

print("%s 個節點的二元樹結構如下：" % n_nodes)
# 迴圈控制敘述逐一印出分類樹模型報表

for i in range(n_nodes):
    if is_leaves[i]:
        print("%snd=%s leaf nd."%(node_depth[i]*" ", i))
    else:
        s1 = "%snd=%s test nd: go to nd %s"
        s2 = " if X[:, %s] <= %s else to nd %s."
        print(''.join([s1, s2])
        % (node_depth[i] * " ",
        i,
        children_left[i],
        feature[i],
        threshold[i],
        children_right[i],
        ))

print()

# 載入Python 語言字串讀寫套件
from io import StringIO
# import pydot
import pydotplus # conda install pydotplus --y (會**自動**安裝2.38版本的graphviz，另外pandas需1.1.3以前的版本)
# 將樹tree 輸出為StringIO 套件的dot_data
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, feature_names=
list(X_train.columns), class_names = ['Not Default', 'Default'],
filled=True, rounded=True) # filled=True, rounded=True

# An alternative way on Windows
# import os
# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/' # 裝到C:/Program Files (x86)/Graphviz2.38/

# dot_data 轉為graph 物件
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph 寫出pdf
graph.write_pdf("credit.pdf")
# graph 寫出png
# graph.write_png('credit.png')
# 載入IPython 的圖片呈現工具類別Image(還有Audio 與Video)
# from IPython.core.display import Image
# from IPython.display import Image # Same as above
# Image(filename='credit.png')
# 或者直接顯示圖形 Show graph immediately
# Image(graph.create_png())

# Calling C5.0 in Python (https://stackoverflow.com/questions/41070087/calling-c5-0-in-python)

#### 5.2.4.2 酒品評點迴歸樹預測
## ------------------------------------------------------------------------
# Python 基本套件與資料集載入
import numpy as np
import pandas as pd
wine = pd.read_csv("./_data/whitewines.csv")
wine.isnull().sum() # 完整無缺

# 檢視變數型別
print(wine.dtypes)

# 葡萄酒資料摘要統計表
print(wine.describe(include='all'))

# 葡萄酒評點分數分佈
ax = wine.quality.hist()
ax.set_xlabel('quality')
ax.set_ylabel('frequency')
fig = ax.get_figure()
# fig.savefig("./_img/quality_hist.png")

# 切分屬性矩陣X 與類別標籤向量y
X = wine.drop(['quality'], axis=1)
y = wine['quality']
# 切分訓練集與測試集
X_train = X[:3750]
X_test = X[3750:]
y_train = y[:3750]
y_test = y[3750:]

# 測試scikit-learn的tree模組可否接受遺缺值
# pd.options.mode.chained_assignment = None # default = 'warn'
# X_train.loc[np.random.choice(X_train.shape[0], 10), 'alcohol'] = np.nan

from sklearn import tree
# 模型定義(未更改預設設定) 與配適
clf = tree.DecisionTreeRegressor()
# 儲存模型clf 參數值字典(因為直接印出會超出邊界)
dicp = clf.get_params()
# 取出字典的鍵，並轉為串列
dic = list(dicp.keys())
# 以字典推導分六次印出模型clf 的參數值
print({key:dicp.get(key) for key in dic[0:int(len(dic)/6)]})

# 第二次列印模型clf 參數值
print({key:dicp.get(key) for key in
dic[int(len(dic)/6):int(2*len(dic)/6)]})

# 第三次列印模型clf 參數值
print({key:dicp.get(key) for key in
dic[int(2*len(dic)/6):int(3*len(dic)/6)]})

# 第四次列印模型clf 參數值
print({key:dicp.get(key) for key in
dic[int(3*len(dic)/6):int(4*len(dic)/6)]})

# 第五次列印模型clf 參數值
print({key:dicp.get(key) for key in
dic[int(4*len(dic)/6):int(5*len(dic)/6)]})

# 第六次列印模型clf 參數值
print({key:dicp.get(key) for key in
dic[int(5*len(dic)/6):int(6*len(dic)/6)]})

# 迴歸樹模型配適
clf = clf.fit(X_train,y_train)
# 節點數過多(2125 個)，顯示節點過度配適
n_nodes = clf.tree_.node_count
print(' 迴歸樹有{0} 節點。'.format(n_nodes))

# 再次宣告空模clf(同上小節更改為R 語言套件{rpart} 的預設值)
clf = tree.DecisionTreeRegressor(max_leaf_nodes = 10,
min_samples_leaf = 7, max_depth= 30)
clf = clf.fit(X_train,y_train)
# 節點數19 個，顯示配適結果改善
n_nodes = clf.tree_.node_count
print(' 迴歸樹有{0} 節點。'.format(n_nodes))

# 預測訓練集酒質分數y_train_pred
y_train_pred = clf.predict(X_train)
# 檢視訓練集酒質分數的實際值分佈與預測值分佈
print(y_train.describe())

# 訓練集酒質預測分佈內縮
print(pd.Series(y_train_pred).describe())

# 預測測試集酒質分數y_test_pred
y_test_pred = clf.predict(X_test)
print(y_test.describe())

# 測試集酒質預測分佈內縮
print(pd.Series(y_test_pred).describe())

# 計算模型績效
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print(' 訓練集MSE: %.3f, 測試集: %.3f' % (
mean_squared_error(y_train, y_train_pred),
mean_squared_error(y_test, y_test_pred)))

print(' 訓練集R^2: %.3f, 測試集R^2: %.3f' % (
r2_score(y_train, y_train_pred),
r2_score(y_test, y_test_pred)))

#### 報表生成與樹模繪圖(TODO)
n_nodes = clf.tree_.node_count
print(' 分類樹有{0} 個節點.'.format(n_nodes))

children_left = clf.tree_.children_left
s1 = ' 各節點的左子節點分別是{0}'
s2 = '\n{1}(-1 表葉子節點沒有子節點)。'
print(''.join([s1, s2]).format(children_left[:9],
children_left[9:]))

children_right = clf.tree_.children_right
s1 = ' 各節點的右子節點分別是{0}'
s2 = '\n{1}(-1 表葉子節點沒有子節點)。'
print(''.join([s1, s2]).format(children_right[:9],
children_right[9:]))

feature = clf.tree_.feature
s1 = ' 各節點分支屬性索引為(-2 表無分支屬性)'
s2 = '\n{0}。'
print(''.join([s1, s2]).format(feature))

threshold = clf.tree_.threshold
s1 = ' 各節點分支屬性門檻值為(-2 表無分支屬性門檻值)'
s2 = '\n{0}\n{1}\n{2}\n{3}。'
print(''.join([s1, s2]).format(threshold[:6],
threshold[6:12], threshold[12:18], threshold[18:]))

# 各節點樹深串列node_depth
node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
# 各節點是否為葉節點的真假值串列
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
# 值組(節點編號, 父節點深度) 形成的堆疊串列, 初始化時只有根節點
stack = [(0, -1)]
# 從堆疊逐一取出資訊產生報表，堆疊最終會變空
while len(stack) > 0:
    node_i, parent_depth = stack.pop()
    # 自己的深度為父節點深度加1
    node_depth[node_i] = parent_depth + 1
    # 如果是測試節點(i.e. 左子節點不等於右子節點)，而非葉節點
    if (children_left[node_i] != children_right[node_i]):
    # 加左分枝節點，分枝節點的父節點深度正是自己的深度
        stack.append((children_left[node_i],parent_depth+1))
    # 加右分枝節點，分枝節點的父節點深度正是自己的深度
        stack.append((children_right[node_i],parent_depth+1))
    else:
    # is_leaves 原預設全為False，最後有True 有False
        is_leaves[node_i] = True

print(" 各節點的深度分別為：{0}".format(node_depth))

print(" 各節點是否為終端節點的真假值分別為：\n{0}\n{1}"
.format(is_leaves[:10], is_leaves[10:]))

print("%s 個節點的二元樹結構如下：" % n_nodes)
# 迴圈控制敘述逐一印出分類樹模型報表

for i in range(n_nodes):
    if is_leaves[i]:
        print("%snd=%s leaf nd."%(node_depth[i]*" ", i))
    else:
        s1 = "%snd=%s test nd: go to nd %s"
        s2 = " if X[:, %s] <= %s else to nd %s."
        print(''.join([s1, s2])
        % (node_depth[i] * " ",
        i,
        children_left[i],
        feature[i],
        threshold[i],
        children_right[i],
        ))

print()

# 載入Python 語言字串讀寫套件
from io import StringIO
# import pydot
import pydotplus # conda install pydotplus --y (會**自動**安裝2.38版本的graphviz，另外pandas需1.1.3以前的版本)
# 將樹tree 輸出為StringIO 套件的dot_data
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, feature_names=
list(X_train.columns), 
filled=True, rounded=True) # filled=True, rounded=True

# An alternative way on Windows
# import os
# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/' # 裝到C:/Program Files (x86)/Graphviz2.38/

# dot_data 轉為graph 物件
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph 寫出pdf
graph.write_pdf("wine.pdf")
# graph 寫出png
# graph.write_png('credit.png')
# 載入IPython 的圖片呈現工具類別Image(還有Audio 與Video)
# from IPython.core.display import Image
# from IPython.display import Image # Same as above
# Image(filename='credit.png')
# 或者直接顯示圖形 Show graph immediately
# Image(graph.create_png())

# Calling C5.0 in Python (https://stackoverflow.com/questions/41070087/calling-c5-0-in-python)


