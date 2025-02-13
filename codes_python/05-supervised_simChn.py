'''
Collated by Prof. Ching-Shih Tsou 邹庆士 教授 (Ph.D.) at the IDS, CICD, and CISD of NTUB (台北商业大学资讯与决策科学研究所暨智能控制与决策研究室教授兼校务永续发展中心主任); at the ME Dept. and CAIDS of MCUT (2020年借调至明志科技大学机械工程系担任特聘教授兼人工智慧暨资料科学研究中心主任两年); the CSQ (2019年起任品质学会大数据品质应用委员会主任委员); the DSBA (2013年创立台湾资料科学与商业应用协会); and the CARS (2012年创立中华R软体学会)
Notes: This code is provided without warranty.
'''

#### 5.2.1.1 手机简讯过滤案例
## ------------------------------------------------------------------------
import pandas as pd
# 读入手机简讯资料集
sms_raw = pd.read_csv("./_data/sms_spam.csv")
# type：垃圾或正常简讯，text：简讯文字内容
print(sms_raw.dtypes)

# type 次数分布，ham 占多数，但未过度不平衡
print(sms_raw['type'].value_counts()/len(sms_raw['type']))

# Python 自然语言处理工具集(Natural Language ToolKit)
import nltk # !conda install nltk --y
# 串列推导完成分词(nltk.download('punkt') at the first time)(list/tuple/dict comprehension, Python很特别的单行回圈语法)
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

# 串列推导完成转小写(Ibiza 变成ibiza, try "TAIWAN".lower())
token_list1 = [[word.lower() for word in doc]
for doc in token_list0] # doc: 各则简讯，word: 各则简讯中的各个字词
print(token_list1[3][1:7])

# 串列推导移除停用词(nltk.download('stopwords') at the first time)
from nltk.corpus import stopwords
# 179 个英语停用字词
print(len(stopwords.words('english')))

# 停用字or 已被移除
token_list2 = [[word for word in doc if word not in stopwords.words('english')] for doc in token_list1]

print(token_list2[3][1:7])

# 串列推导移除标点符号
import string
token_list3 = [[word for word in doc if word not in
string.punctuation] for doc in token_list2]
print(token_list3[3][1:7])

# 串列推导移除所有数字(4 不见了)
token_list4 = [[word for word in doc if not word.isdigit()]
for doc in token_list3]
print(token_list4[3][1:7])

# 三层巢状串列推导移除字符中夹杂数字或标点符号的情形
token_list5 = [[''.join([i for i in word if not i.isdigit()
and i not in string.punctuation]) for word in doc]
for doc in token_list4] # doc: 各则简讯，word: 各则简讯中的各个字词，i: 各个字词中的各个字元
# similar to paste() in R
# £10,000 变成£
print(token_list5[3][1:7])

# 串列推导移除空元素
token_list6 =[list(filter(None, doc)) for doc in token_list5]
print(token_list6[3][1:7])

# 载入nltk.stem 的WordNet 词形还原库(nltk.download('wordnet') nltk.download('omw-1.4') at the first time)
from nltk.stem import WordNetLemmatizer
# 宣告词形还原器
lemma = WordNetLemmatizer()
# 串列推导完成词形还原(needs 变成need)
token_list6 = [[lemma.lemmatize(word) for word in doc]
for doc in token_list6]
print(token_list6[3][1:7])

# 串列推导完成各则字词的串接
# join() 方法将各则简讯doc 中分开的字符又连接起来
token_list7 = [' '.join(doc) for doc in token_list6]
print(token_list7[:2])

import pandas as pd
# 从feature_extraction 模组载入词频计算与DTM 建构类别
from sklearn.feature_extraction.text import CountVectorizer # 计算词频，将各则简讯向量化
# 宣告空模
vec = CountVectorizer() # binary: bool, default=False. If True, all non zero counts are set to 1.
# 传入简讯配适实模并转换为DTM 稀疏矩阵X
X = vec.fit_transform(token_list7) # Try token_list6 and you will get an error !
dir(vec)

# scipy 套件稀疏矩阵类别(csr, compressed sparse rows)
print(type(X))

# 稀疏矩阵储存词频的方式：(横，纵) 词频
print(X[:2]) # 前两则简讯的词频在稀疏矩阵中的存放方式

# import sys
# import numpy
# numpy.set_printoptions(threshold=sys.maxsize)
# print(X.toarray()[:2]) # 转成常规矩阵后，方可见前两则简讯的完整词频向量
dir(X)

# X 转为常规矩阵(X.toarray())，并组织为pandas 资料框
sms_dtm = pd.DataFrame(X.toarray(),
columns=vec.get_feature_names()) # vec.get_feature_names_out()
# sms_dtm.max(axis=0).max() # 不是0-1编码(Deep Learning)，是词频编码(Bag of Word, BoW)
# 5559 列(则)7484 行(字) 的结构
print(sms_dtm.shape)

# 模型vec 取出DTM 各字词的get_feature_names() 方法
print(len(vec.get_feature_names())) # 共有7484 个字词 vec.get_feature_names_out()

print(vec.get_feature_names()[300:305]) # vec.get_feature_names_out()

import numpy as np # 书中遗漏此列程式码

# 5559 则简讯中app 此字只有6 则正词频，的确稀疏(新版numpy请用下行注解程式码)
print(np.argwhere(sms_dtm['app'] > 0)) # 列向量
# print(np.argwhere((sms_dtm['app'] > 0).values.reshape((-1,1)))) # 新版numpy需转成行向量

# DTM 部分内容
print(sms_dtm.iloc[4460:4470, 300:305])
# sms_dtm.max().max() # 15, 原始词频dtm，适合配适multinomialNB()

# 训练与测试集切分(sms_raw, sms_dtm, token_list6)
sms_raw_train = sms_raw.iloc[:4170, :]
sms_raw_test = sms_raw.iloc[4170:, :]
sms_dtm_train = sms_dtm.iloc[:4170, :]
sms_dtm_test = sms_dtm.iloc[4170:, :]
token_list6_train = token_list6[:4170]
token_list6_test = token_list6[4170:]
# 查核各子集类别分布
print(sms_raw_train['type'].value_counts()/
len(sms_raw_train['type']))

print(sms_raw_test['type'].value_counts()/
len(sms_raw_test['type']))

# WordCloud() 统计词频须跨篇组合所有词项
tokens_train = [token for doc in token_list6_train
for token in doc]
print(len(tokens_train))

# 逻辑值索引结合zip() 捆绑函数，再加判断句与串列推导
tokens_train_spam = [token for is_spam, doc in
zip(sms_raw_train['type'] == 'spam' , token_list6_train)
if is_spam for token in doc]
# 取出正常简讯
tokens_train_ham = [token for is_ham, doc in
zip(sms_raw_train['type'] == 'ham' , token_list6_train)
if is_ham for token in doc]
# 逗号接合训练与spam 和ham 两子集tokens
str_train = ','.join(tokens_train)
str_train_spam = ','.join(tokens_train_spam)
str_train_ham = ','.join(tokens_train_ham)

# Python 文字云套件(conda install -c conda-forge wordcloud --y)
from wordcloud import WordCloud
# 宣告文字云物件(最大字数max_words 预设为200)
wc_train = WordCloud(background_color="white",
prefer_horizontal=0.5)
# 传入资料统计，并产制文字云物件
wc_train.generate(str_train) # str_train -> str_train_spam, str_train_ham
# 呼叫matplotlib.pyplot 模组下的imshow() 方法绘图
import matplotlib.pyplot as plt
plt.imshow(wc_train)
plt.axis("off")
# plt.show()
# plt.savefig('wc_train.png')
# 限于篇幅，str_train_spam 和str_train_ham 文字云绘制代码省略

# 载入多项式天真贝氏模型类别
from sklearn.naive_bayes import MultinomialNB
# 模型定义、配适与预测
clf = MultinomialNB()

clf.fit(sms_dtm_train, sms_raw_train['type'])
train = clf.predict(sms_dtm_train)
print(" 训练集正确率为{}".format(sum(sms_raw_train['type'] ==
train)/len(train)))

pred = clf.predict(sms_dtm_test)
print(" 测试集正确率为{}".format(sum(sms_raw_test['type'] ==
pred)/len(pred)))
# dir(clf)

# 训练所用的各类样本数
print(clf.class_count_)

# 两类与7612(7484) 个属性的交叉列表**(绝对频次表)**
print(clf.feature_count_)

print(clf.feature_count_.shape)

# 已知类别下，各属性之条件机率(似然率)Pr[x_i|y] 的对数值**(相对频次表再取对数值)**
print(clf.feature_log_prob_[:, :4])

print(clf.feature_log_prob_.shape)

# 将对数条件机率转成机率值(补充程式码)
feature_prob = np.exp(clf.feature_log_prob_)
print(feature_prob.shape)
print(feature_prob[:, :4])
# 验证两类之机率值总和为1(补充程式码)
print(np.apply_along_axis(np.sum, 1, feature_prob)) # [1. 1.]
# 两类最大字词机率值(补充程式码)
print(np.apply_along_axis(np.max, 1, feature_prob)) # [0.00813987 0.01839848]]
# 抓出两类机率前五高的字词，与文字云结果相符(补充程式码)
print(sms_dtm.columns.values[np.argsort(-feature_prob)[:,:5]])
# ham: ['nt' 'get' 'go' 'ok' 'call']
# spam: ['call' 'free' 'txt' 'mobile' 'text']

import numpy as np
# 载入sklearn 交叉验证模型选择的重要函数
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import sem
# 自定义k 折交叉验证模型绩效计算函数
def evaluate_cross_validation(clf, X, y, K): # 输入全样本集
    # 创建k 折交叉验证迭代器(iterator)，用于X 与y 的切分
    cv = KFold(n_splits=K, shuffle=True, random_state=0)
    scores = cross_val_score(clf, X, y, cv=cv)
    print("{}折交叉验证结果如下：\n{}".format(K, scores))
    tmp = " 平均正确率：{0:.3f}(+/-标准误{1:.3f})"
    print(tmp.format(np.mean(scores), sem(scores)))

evaluate_cross_validation(clf, sms_dtm, sms_raw['type'], 5)

#### 5.2.2.1 电离层无线电讯号案例
## ------------------------------------------------------------------------
import numpy as np
import pandas as pd
iono = pd.read_csv("./_data/ionosphere.data", header=None)
# 切分属性矩阵与目标向量
X = iono.iloc[:, :-1] # 纵向前包后不包
y = iono.iloc[:, -1] # 纵向无冒号，故无所谓前包后不包
print(X.shape)

print(y.shape)

# 无名目属性，适合k 近邻学习
print(X.dtypes)

# 资料无遗缺，可直接进行k 近邻学习
print(" 遗缺{}个数值".format(X.isnull().sum().sum()))
# print(" 各变量遗缺概况：")
# print(X.isnull().sum())

# 训练集与测试集切分
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
random_state=14) # random_state=1234
print(" 训练集有{}样本".format(X_train.shape[0]))

print(" 测试集有{}样本".format(X_test.shape[0]))

print(" 每个样本有{}属性".format(X_train.shape[1]))

print(" 资料集类别分布为：\n{}.".format(y.value_counts()/len(y)))

print(" 训练集类别分布为：\n{}."
.format(y_train.value_counts()/len(y_train)))

print(" 测试集类别分布为：\n{}."
.format(y_test.value_counts()/len(y_test)))

# 载入sklearn 前处理模组的标准化转换类别
from sklearn.preprocessing import StandardScaler
# 模型定义(未更改预设设定)、配适与转换(似乎多余！)
sc = StandardScaler()
# 配适与转换接续执行函数fit_transform()
X_train_std = sc.fit_transform(X_train)
# 依训练集拟合的(**标准化**)模型，对测试集做转换
X_test_std = sc.transform(X_test)
# sc.mean_ from training set
# sc.var_ from training set

# 整个属性矩阵标准化是为了交叉验证调参(注意！模型sc 内容会变)
X_std = sc.fit_transform(X)
# sc.mean_ from whole dataset
# sc.var_ from whole dataset

# 建议比较sc_train = StandardScaler(), sc_train.fit_transform(X_train)以及sc = StandardScaler(), sc.fit_transform(X)两者sc_train.mean_和sc.mean_以及sc_train.var_和sc.var_的差异！！！

# 载入sklearn 近邻学习模组的k 近邻分类类别
from sklearn.neighbors import KNeighborsClassifier
# 模型定义(未更改预设设定)、配适与转换
estimator = KNeighborsClassifier()
estimator.fit(X_train_std, y_train)
# 模型estimator 的get_params() 方法取出模型参数：
# Minkowski 距离之p 为2(欧几里德距离) 与邻居数是5
for name in ['metric','n_neighbors','p']:
    print(estimator.get_params()[name])

# 对训练集进行预测
train_pred = estimator.predict(X_train_std)
# 训练集前五笔预测值
print(train_pred[:5])

# 训练集前五笔实际值
print(y_train[:5])

train_acc = np.mean(y_train == train_pred) * 100
print(" 训练集正确率为{0:.1f}%".format(train_acc))

# 对测试集进行预测
y_pred = estimator.predict(X_test_std)
# 测试集前五笔预测值
print(y_pred[:5])

# 测试集前五笔实际值
print(y_test[:5])

test_acc = np.mean(y_test == y_pred) * 100
print(" 测试集正确率为{0:.1f}%".format(test_acc))
# 以上是单次保留法的结果，重复多次的保留法更好！

# sklearn 套件中模型选择模组下交叉验证训练测试机制之绩效计算函数
from sklearn.model_selection import cross_val_score
# 预设为三折(新版sklearn预设改为五折)交叉验证运行一次
scores = cross_val_score(estimator, X_std, y,
scoring='accuracy')
print(scores.shape)

average_accuracy = np.mean(scores) * 100
print(" 五次的平均正确率为{0:.1f}%".format(average_accuracy))

# std_accuracy = np.std(scores, ddof=1) * 100 # 留意np.std()是计算母体标准差
# print(" 五次的正确率标准差为{0:.1f}%".format(std_accuracy))

# 逐步收纳结果用
avg_scores = []
all_scores = []
# 定义待调参数候选集
parameter_values = list(range(1, 32, 3)) # range(1, 31)
# 对每一参数候选值，执行下方内缩叙述
for n_neighbors in parameter_values:
    # 宣告模型规格n_neighbors
    estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
    # cross_val_score() 依模型规格与资料集进行交叉验证训练和测试
    sc=cross_val_score(estimator,X_std,y,scoring='accuracy')
    # 绩效分数(accuracy) 平均值计算与添加
    avg_scores.append(np.mean(sc))
    all_scores.append(sc)
# 近邻数从1 到20 的平均正确率
print(len(avg_scores))

print(avg_scores)

# 近邻数从1 到20 的五折交叉验证结果
print(len(all_scores))

# 不同近邻数k 值下，五次交叉验证的正确率
print(all_scores[:4])

# 不同近邻数下平均正确率折线图
from matplotlib import pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
plt.xticks(np.arange(1, 32, 3)) # np.arange(1, 21)
ax.plot(parameter_values, avg_scores, '-o')
# fig.savefig('./_img/iono_tuning_avg_scores.png')

# 载入sklearn 前处理模组正规化转换类别
from sklearn.preprocessing import MinMaxScaler
# 载入统计机器学习流程化模组
from sklearn.pipeline import Pipeline
# 流程定义
pipe = Pipeline([('scale', MinMaxScaler()), ('predict',
KNeighborsClassifier())])
# 流程与资料传入cross_val_score() 函数
scores = cross_val_score(pipe, X, y, scoring='accuracy')
# 五折交叉验证结果
print(" 五次正确率结果为{}%".format(scores*100))

print(" 平均正确率为{0:.1f}%".format(np.mean(scores) * 100))

print(" 五次正确率的标准差为{0:.1f}%".format(np.std(scores, ddof=1) * 100))

#### 5.2.3.1 光学手写字元案例
## ------------------------------------------------------------------------
import pandas as pd
letters = pd.read_csv("./_data/letterdata.csv")
# 检视变数型别
print(letters.dtypes)

# 各整数值变数介于0 到15 之间(4 bits 像素值)
print(letters.describe(include = 'all'))

# 目标变数各类别分布平均(预设依各类频次降幂排序)
print(letters['letter'].value_counts()) # np.unique(letters.letter, return_counts=True)

# 载入sklearn 属性挑选模组的变异数过滤类别
from sklearn.feature_selection import VarianceThreshold
# 模型定义、配适与转换(i.e. 删除零变异属性)
vt = VarianceThreshold(threshold=0)
# 并无发现零变异属性
print(vt.fit_transform(letters.iloc[:,1:]).shape)

# 没有超过(低于或等于) 变异数门槛值0 的属性是0 个
import numpy as np
print(np.sum(vt.get_support() == False)) # Get a mask of the features selected
# vt.get_support(indices=True) # Get integer index of the features selected

# 计算相关系数方阵后转numpy ndarray
cor = letters.iloc[:,1:].corr().values
print(cor[:5,:5])

# 相关系数超标(+-0.8) 真假值方阵
import numpy as np
np.fill_diagonal(cor, 0) # 变更对角线元素值为0
threTF = abs(cor) > 0.8
print(threTF[:5,:5])

# 类似R 语言的which(真假值矩阵, arr.ind=TRUE)
print(np.argwhere(threTF == True))

# 核对变数名称，注意相关系数计算时已排除掉第1 个变数letter
print(letters.columns[1:5])

# pandas 资料框boxplot() 方法绘制并排盒须图(26组数据)
ax1 = letters[['xbox', 'letter']].boxplot(by = 'letter')
fig1 = ax1.get_figure()
# fig1.savefig('./_img/xbox_boxplot.png')
ax2 = letters[['ybar', 'letter']].boxplot(by = 'letter')
fig2 = ax2.get_figure()
# fig2.savefig('./_img/ybar_boxplot.png')

# 训练与测试集切分
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
letters.iloc[:, 1:], letters['letter'], test_size=0.2,
random_state=0)
# 数据标准化
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# 计算X_train 各变数的mu 和sigma
sc.fit(X_train)
# 真正做转换
X_train_std = sc.transform(X_train)
# 以X_train 各变数的mu 和sigma 对X_test 做转换
X_test_std = sc.transform(X_test)

# SVC: 支援向量分类(Support Vector Classification)
# SVR: 支援向量回归(Support Vector Regression)
# OneClassSVM: 非监督式离群侦测(Outlier Detection)
from sklearn.svm import SVC
# 模型定义(使用线性核函数，先前预设的核函数为'linear'，现已改为'rbf')、配适与转换
svm = SVC(kernel='poly') # 原来括弧内是空白
svm.fit(X_train_std, y_train)
tr_pred = svm.predict(X_train_std)
y_pred = svm.predict(X_test_std)
# 训练集前5 笔预测值
print(tr_pred[:5])

# 训练集前5 笔实际值
print(y_train[:5].tolist())

# 测试集前5 笔预测值
print(y_pred[:5])

# 测试集前5 笔实际值
print(y_test[:5].tolist())

# 注意Python 另一种输出格式化语法(% 符号)
err_tr = (y_train != tr_pred).sum()/len(y_train)
print(' 训练集错误率为：%.5f' % err_tr)

# 测试集错误率稍高于训练集的错误率
err = (y_test != y_pred).sum()/len(y_test)
print(' 测试集错误率为：%.5f' % err)

# 变更径向基底函数之参数gamma 为0.2，(5.34) 式的C 为1.0
svm = SVC(kernel='rbf', random_state=0, gamma=0.2, C=1.0) # You can try SVC().
svm.fit(X_train_std, y_train)
tr_pred = svm.predict(X_train_std)
y_pred = svm.predict(X_test_std)
# 训练集前5 笔预测值
print(tr_pred[:5])

# 训练集前5 笔实际值
print(y_train[:5].tolist())

# 测试集前5 笔预测值
print(y_pred[:5])

# 测试集前5 笔实际值
print(y_test[:5].tolist())

err_tr = (y_train.values != tr_pred).sum()/len(y_train)
print(' 训练集错误率为：%.5f' % err_tr)

# 测试集错误率也是稍高于训练集的错误率
err = (y_test != y_pred).sum()/len(y_test)
print(' 测试集错误率为：%.5f' % err)

# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)

# 载入整合pandas, scikit-learn 与xgboost 的套件pandas_ml(请注意scikit-learn须降版为0.21.1，pandas须降为0.22.0，最好是另创一个虚拟环境)
# conda install -c conda-forge scikit-learn==0.21.1 pandas==0.22.0 pandas_ml --y
import pandas_ml as pdml
# 注意！须传入numpy ndarray 物件，以生成正确的混淆矩阵
cm = pdml.ConfusionMatrix(y_test.values, y_pred)
# dir(cm)
# 混淆矩阵转成pandas 资料框，方便书中结果呈现
cm_df = cm.to_dataframe(normalized=False, calc_sum=True,
sum_label='all')
# 混淆矩阵部分结果
print(cm_df.iloc[:12, :12])

# stats() 方法生成整体(3.2.2.3 节) 与类别相关指标(3.2.2.4 节)
perf_indx = cm.stats()
# 储存为collections 套件的有序字典结构(OrderedDict)
print(type(perf_indx))

# 有序字典结构的键，其中cm 为相同的混淆矩阵
print(perf_indx.keys())

# overall 键下也是有序字典结构
print(type(perf_indx['overall']))
# perf_indx['overall'].keys()

# 整体指标内容如下：
print(" 分类模型正确率为：{}".format(perf_indx['overall']
['Accuracy']))

print(" 正确率95% 信赖区间为：\n{}".format(perf_indx
['overall']['95% CI']))

print("Kappa 统计量为：\n{}".format(perf_indx['overall']
['Kappa']))

# class 键下是pandas 资料框结构
print(type(perf_indx['class']))

# 26 个字母(纵向) 各有26 个类别(横向) 相关指标
print(perf_indx['class'].shape)

print(perf_indx['class'])

# 混淆矩阵热图视觉化，请读者自行尝试
import matplotlib.pyplot as plt
ax = cm.plot()
fig = ax.get_figure()
# fig.savefig('./_img/svc_rbf.png')

#### 5.2.4.1 银行贷款风险管理案例
## ------------------------------------------------------------------------
import numpy as np
import pandas as pd
# 读入UCI 授信客户资料集
credit = pd.read_csv("./_data/germancredit.csv")
print(credit.shape)

# 检视变数型别
print(credit.dtypes)

# 目标变数Default(已为0-1 值) 次数分布
print(credit.Default.value_counts())

# 变数转换字典target
target = {0: "Not Default", 1: "Default"}
credit.Default = credit.Default.map(target)

# 成批产制类别变数(dtype 为object) 的次数分布表(存为字典结构)
# 先以逻辑值索引取出object 栏位名称
col_cat = credit.columns[credit.dtypes == "object"]
# 逐步收纳各类别变数次数统计结果用
counts_dict = {}
# 取出各栏类别值统计频次
for col in col_cat:
    counts_dict[col] = credit[col].value_counts()
# 印出各类别变数次数分布表
print(counts_dict)

# 代号与易了解名称对照字典
print(dict(zip(credit.checkingstatus1.unique(),["< 0 DM",
"0-200 DM","no account","> 200 DM"])))

# 逐栏转换易了解的类别名称
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

# 基于篇幅考量，上面只显示前五类别变数的转换代码，以下请读者自行执行。
credit.status = credit.status.map(dict(zip(credit.status.unique(),["M/Div/Sep","F/Div/Sep/Mar","M/Single","M/Mar/Wid"])))
credit.others = credit.others.map(dict(zip(credit.others.unique(),["none","co-applicant","guarantor"])))
credit.property = credit.property.map(dict(zip(credit.property.unique(),["none","co_applicant","guarantor"])))
credit.otherplans = credit.otherplans.map(dict(zip(credit.otherplans.unique(),["bank","stores","none"])))
credit['rent'] = credit['housing'] == "A151"
credit['rent'].value_counts()
# del credit['housing']

credit.job = credit.job.map(dict(zip(credit.job.unique(),["unemployed","unskilled","skilled", "mgt/self-employed"])))
credit.tele = credit.tele.map(dict(zip(credit.tele.unique(),["none","yes"])))
credit.foreign = credit.foreign.map(dict(zip(credit.foreign.unique(),["foreign","german"])))
    
# 资料表内容较容易了解
print(credit.head())

# 授信客户资料摘要统计表
print(credit.describe(include='all'))
# credit.isnull().sum() # 除了'property'外均完整
# crosstab() 函数建支票存款帐户状况，与是否违约的二维列联表
ck_f = pd.crosstab(credit['checkingstatus1'],
credit['Default'], margins=True)
# 计算相对次数
ck_f.Default = ck_f.Default/ck_f.All
ck_f['Not Default'] = ck_f['Not Default']/ck_f.All
print(ck_f)

# 储蓄存款帐户余额状况，与是否违约的二维列联表
sv_f = pd.crosstab(credit['savings'],
credit['Default'], margins=True)
sv_f.Default = sv_f.Default/sv_f.All

sv_f['Not Default'] = sv_f['Not Default']/sv_f.All
print(sv_f)

# 与R 语言summary() 输出相比，多了样本数count 与标准差std
print(credit['duration'].describe())

print(credit['amount'].describe())

# 字串转回0-1 整数值
inv_target = {"Not Default": 0, "Default": 1}
credit.Default = credit.Default.map(inv_target)

# 成批完成类别预测变数标签编码 (Python自己来，R自动完成)
from sklearn.preprocessing import LabelEncoder
# 先以逻辑值索引取出类别栏位名称
col_cat = credit.columns[credit.dtypes == "object"]
# 宣告空模
le = LabelEncoder()
# 逐栏取出类别变数值后进行标签编码
for col in col_cat:
    credit[col] = le.fit_transform(credit[col].astype(str))
# credit.property.value_counts()
# 切分类别标签向量y 与属性矩阵X
y = credit['Default']
X = credit.drop(['Default'], axis=1)
# 切分训练集及测试集，random_state 引数设定乱数种子
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.1, random_state=33)

# 训练集类别标签次数分布表
Default_train = pd.DataFrame(y_train.value_counts(sort =
True))
# 计算与建立累积和栏位'cum_sum'
Default_train['cum_sum'] = Default_train['Default'].cumsum()
# 计算与建立相对次数栏位'perc'
tot = len(y_train)
Default_train['perc']=100*Default_train['Default']/tot
# 计算与建立累积相对次数栏位'cum_perc'
Default_train['cum_perc']=100*Default_train['cum_sum']/tot

### 限于篇幅，书本省略下面程式码
# 测试集类别标签次数分布表
Default_test = pd.DataFrame(y_test.value_counts(sort=True))
# 计算与建立累积和栏位'cum_sum'
Default_test['cum_sum'] = Default_test['Default'].cumsum()
# 计算与建立相对次数栏位'perc'
Default_test['perc'] = 100*Default_test['Default']/len(y_test)
# 计算与建立累积相对次数栏位'cum_perc'
Default_test['cum_perc'] = 100*Default_test['cum_sum']/len(y_test)
###

# 比较训练集与测试集类别标签分布
print(Default_train)
print(Default_test)

# 载入sklearn 套件的树状模型模组tree
from sklearn import tree
# 宣告DecisionTreeClassifier() 类别空模clf(未更改预设设定)
clf = tree.DecisionTreeClassifier()
# 传入训练资料拟合实模clf
clf = clf.fit(X_train, y_train)
# ValueError: could not convert string to float: '> 200 DM' (前面如果字串自变数未标签编码，则会报错！)
# 预测训练集标签train_pred
train_pred = clf.predict(X_train)
print(' 训练集错误率为{0}.'.format(np.mean(y_train !=
train_pred)))

# 预测测试集标签test_pred
test_pred = clf.predict(X_test)
# 训练集错误率远低于测试集，过度配适的症兆
print(' 测试集错误率为{0}.'.format(np.mean(y_test !=
test_pred)))
print('此树有{}节点'.format(clf.tree_.node_count)) # 好复杂的一棵树！所以接下来我们调参
print(clf.get_params())
keys = ['max_depth', 'max_leaf_nodes', 'min_samples_leaf']
# type(clf.get_params()) # dict
print([(key, clf.get_params().get(key)) for key in keys])

# 再次宣告空模clf(更改上述三参数设定)、配适与预测
clf = tree.DecisionTreeClassifier(max_leaf_nodes = 10,
min_samples_leaf = 7, max_depth= 30)
clf = clf.fit(X_train,y_train)
train_pred = clf.predict(X_train)
print(' 训练集错误率为{0}.'.format(np.mean(y_train !=
train_pred)))

# 过度配适情况已经改善
test_pred = clf.predict(X_test)
print(' 测试集错误率为{0}.'.format(np.mean(y_test !=
test_pred)))

#### 报表生成与树模绘图
n_nodes = clf.tree_.node_count
print(' 分类树有{0} 个节点.'.format(n_nodes))

children_left = clf.tree_.children_left
s1 = ' 各节点的左子节点分别是{0}'
s2 = '\n{1}(-1 表叶子节点没有子节点)。'
print(''.join([s1, s2]).format(children_left[:9],
children_left[9:]))

children_right = clf.tree_.children_right
s1 = ' 各节点的右子节点分别是{0}'
s2 = '\n{1}(-1 表叶子节点没有子节点)。'
print(''.join([s1, s2]).format(children_right[:9],
children_right[9:]))

feature = clf.tree_.feature
s1 = ' 各节点分支属性索引为(-2 表无分支属性)'
s2 = '\n{0}。'
print(''.join([s1, s2]).format(feature))

threshold = clf.tree_.threshold
s1 = ' 各节点分支属性门槛值为(-2 表无分支属性门槛值)'
s2 = '\n{0}\n{1}\n{2}\n{3}。'
print(''.join([s1, s2]).format(threshold[:6],
threshold[6:12], threshold[12:18], threshold[18:]))

# 各节点树深串列node_depth
node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
# 各节点是否为叶节点的真假值串列
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
# 值组(节点编号, 父节点深度) 形成的堆叠串列, 初始化时只有根节点
stack = [(0, -1)]
# 从堆叠逐一取出资讯产生报表，堆叠最终会变空
while len(stack) > 0:
    node_i, parent_depth = stack.pop()
    # 自己的深度为父节点深度加1
    node_depth[node_i] = parent_depth + 1
    # 如果是测试节点(i.e. 左子节点不等于右子节点)，而非叶节点
    if (children_left[node_i] != children_right[node_i]):
    # 加左分枝节点，分枝节点的父节点深度正是自己的深度
        stack.append((children_left[node_i],parent_depth+1))
    # 加右分枝节点，分枝节点的父节点深度正是自己的深度
        stack.append((children_right[node_i],parent_depth+1))
    else:
    # is_leaves 原预设全为False，最后有True 有False
        is_leaves[node_i] = True

print(" 各节点的深度分别为：{0}".format(node_depth))

print(" 各节点是否为终端节点的真假值分别为：\n{0}\n{1}"
.format(is_leaves[:10], is_leaves[10:]))

print("%s 个节点的二元树结构如下：" % n_nodes)
# 回圈控制叙述逐一印出分类树模型报表

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

# 载入Python 语言字串读写套件
from io import StringIO
# import pydot
import pydotplus # conda install pydotplus --y (会**自动**安装2.38版本的graphviz，另外pandas需1.1.3以前的版本)
# 将树tree 输出为StringIO 套件的dot_data
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, feature_names=
list(X_train.columns), class_names = ['Not Default', 'Default'],
filled=True, rounded=True) # filled=True, rounded=True

# An alternative way on Windows
# import os
# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/' # 装到C:/Program Files (x86)/Graphviz2.38/

# dot_data 转为graph 物件
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph 写出pdf
graph.write_pdf("credit.pdf")
# graph 写出png
# graph.write_png('credit.png')
# 载入IPython 的图片呈现工具类别Image(还有Audio 与Video)
# from IPython.core.display import Image
# from IPython.display import Image # Same as above
# Image(filename='credit.png')
# 或者直接显示图形 Show graph immediately
# Image(graph.create_png())

# Calling C5.0 in Python (https://stackoverflow.com/questions/41070087/calling-c5-0-in-python)

#### 5.2.4.2 酒品评点回归树预测
## ------------------------------------------------------------------------
# Python 基本套件与资料集载入
import numpy as np
import pandas as pd
wine = pd.read_csv("./_data/whitewines.csv")
wine.isnull().sum() # 完整无缺

# 检视变数型别
print(wine.dtypes)

# 葡萄酒资料摘要统计表
print(wine.describe(include='all'))

# 葡萄酒评点分数分布
ax = wine.quality.hist()
ax.set_xlabel('quality')
ax.set_ylabel('frequency')
fig = ax.get_figure()
# fig.savefig("./_img/quality_hist.png")

# 切分属性矩阵X 与类别标签向量y
X = wine.drop(['quality'], axis=1)
y = wine['quality']
# 切分训练集与测试集
X_train = X[:3750]
X_test = X[3750:]
y_train = y[:3750]
y_test = y[3750:]

# 测试scikit-learn的tree模组可否接受遗缺值
# pd.options.mode.chained_assignment = None # default = 'warn'
# X_train.loc[np.random.choice(X_train.shape[0], 10), 'alcohol'] = np.nan

from sklearn import tree
# 模型定义(未更改预设设定) 与配适
clf = tree.DecisionTreeRegressor()
# 储存模型clf 参数值字典(因为直接印出会超出边界)
dicp = clf.get_params()
# 取出字典的键，并转为串列
dic = list(dicp.keys())
# 以字典推导分六次印出模型clf 的参数值
print({key:dicp.get(key) for key in dic[0:int(len(dic)/6)]})

# 第二次列印模型clf 参数值
print({key:dicp.get(key) for key in
dic[int(len(dic)/6):int(2*len(dic)/6)]})

# 第三次列印模型clf 参数值
print({key:dicp.get(key) for key in
dic[int(2*len(dic)/6):int(3*len(dic)/6)]})

# 第四次列印模型clf 参数值
print({key:dicp.get(key) for key in
dic[int(3*len(dic)/6):int(4*len(dic)/6)]})

# 第五次列印模型clf 参数值
print({key:dicp.get(key) for key in
dic[int(4*len(dic)/6):int(5*len(dic)/6)]})

# 第六次列印模型clf 参数值
print({key:dicp.get(key) for key in
dic[int(5*len(dic)/6):int(6*len(dic)/6)]})

# 回归树模型配适
clf = clf.fit(X_train,y_train)
# 节点数过多(2125 个)，显示节点过度配适
n_nodes = clf.tree_.node_count
print(' 回归树有{0} 节点。'.format(n_nodes))

# 再次宣告空模clf(同上小节更改为R 语言套件{rpart} 的预设值)
clf = tree.DecisionTreeRegressor(max_leaf_nodes = 10,
min_samples_leaf = 7, max_depth= 30)
clf = clf.fit(X_train,y_train)
# 节点数19 个，显示配适结果改善
n_nodes = clf.tree_.node_count
print(' 回归树有{0} 节点。'.format(n_nodes))

# 预测训练集酒质分数y_train_pred
y_train_pred = clf.predict(X_train)
# 检视训练集酒质分数的实际值分布与预测值分布
print(y_train.describe())

# 训练集酒质预测分布内缩
print(pd.Series(y_train_pred).describe())

# 预测测试集酒质分数y_test_pred
y_test_pred = clf.predict(X_test)
print(y_test.describe())

# 测试集酒质预测分布内缩
print(pd.Series(y_test_pred).describe())

# 计算模型绩效
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print(' 训练集MSE: %.3f, 测试集: %.3f' % (
mean_squared_error(y_train, y_train_pred),
mean_squared_error(y_test, y_test_pred)))

print(' 训练集R^2: %.3f, 测试集R^2: %.3f' % (
r2_score(y_train, y_train_pred),
r2_score(y_test, y_test_pred)))

#### 报表生成与树模绘图(TODO)
n_nodes = clf.tree_.node_count
print(' 分类树有{0} 个节点.'.format(n_nodes))

children_left = clf.tree_.children_left
s1 = ' 各节点的左子节点分别是{0}'
s2 = '\n{1}(-1 表叶子节点没有子节点)。'
print(''.join([s1, s2]).format(children_left[:9],
children_left[9:]))

children_right = clf.tree_.children_right
s1 = ' 各节点的右子节点分别是{0}'
s2 = '\n{1}(-1 表叶子节点没有子节点)。'
print(''.join([s1, s2]).format(children_right[:9],
children_right[9:]))

feature = clf.tree_.feature
s1 = ' 各节点分支属性索引为(-2 表无分支属性)'
s2 = '\n{0}。'
print(''.join([s1, s2]).format(feature))

threshold = clf.tree_.threshold
s1 = ' 各节点分支属性门槛值为(-2 表无分支属性门槛值)'
s2 = '\n{0}\n{1}\n{2}\n{3}。'
print(''.join([s1, s2]).format(threshold[:6],
threshold[6:12], threshold[12:18], threshold[18:]))

# 各节点树深串列node_depth
node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
# 各节点是否为叶节点的真假值串列
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
# 值组(节点编号, 父节点深度) 形成的堆叠串列, 初始化时只有根节点
stack = [(0, -1)]
# 从堆叠逐一取出资讯产生报表，堆叠最终会变空
while len(stack) > 0:
    node_i, parent_depth = stack.pop()
    # 自己的深度为父节点深度加1
    node_depth[node_i] = parent_depth + 1
    # 如果是测试节点(i.e. 左子节点不等于右子节点)，而非叶节点
    if (children_left[node_i] != children_right[node_i]):
    # 加左分枝节点，分枝节点的父节点深度正是自己的深度
        stack.append((children_left[node_i],parent_depth+1))
    # 加右分枝节点，分枝节点的父节点深度正是自己的深度
        stack.append((children_right[node_i],parent_depth+1))
    else:
    # is_leaves 原预设全为False，最后有True 有False
        is_leaves[node_i] = True

print(" 各节点的深度分别为：{0}".format(node_depth))

print(" 各节点是否为终端节点的真假值分别为：\n{0}\n{1}"
.format(is_leaves[:10], is_leaves[10:]))

print("%s 个节点的二元树结构如下：" % n_nodes)
# 回圈控制叙述逐一印出分类树模型报表

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

# 载入Python 语言字串读写套件
from io import StringIO
# import pydot
import pydotplus # conda install pydotplus --y (会**自动**安装2.38版本的graphviz，另外pandas需1.1.3以前的版本)
# 将树tree 输出为StringIO 套件的dot_data
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, feature_names=
list(X_train.columns), 
filled=True, rounded=True) # filled=True, rounded=True

# An alternative way on Windows
# import os
# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/' # 装到C:/Program Files (x86)/Graphviz2.38/

# dot_data 转为graph 物件
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph 写出pdf
graph.write_pdf("wine.pdf")
# graph 写出png
# graph.write_png('credit.png')
# 载入IPython 的图片呈现工具类别Image(还有Audio 与Video)
# from IPython.core.display import Image
# from IPython.display import Image # Same as above
# Image(filename='credit.png')
# 或者直接显示图形 Show graph immediately
# Image(graph.create_png())

# Calling C5.0 in Python (https://stackoverflow.com/questions/41070087/calling-c5-0-in-python)


