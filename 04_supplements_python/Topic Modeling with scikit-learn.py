'''
Collated by Ching-Shih Tsou 鄒慶士 博士 (Ph.D.) Distinguished Prof. at the Department of Mechanical Engineering/Director at the Center of Artificial Intelligence & Data Science (機械工程系特聘教授兼人工智慧暨資料科學研究中心主任), MCUT (明志科技大學); Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), NTUB (國立臺北商業大學)
Notes: This code is provided without warranty.
'''


#### 資料載入與預處理 Load and Pre-process Data

import os
import numpy as np
import pandas as pd

DATA_PATH = 'nipstxt/'
print(os.listdir(DATA_PATH))

folders = ["nips{0:02}".format(i) for i in range(0,13)] # DATA_PATH下的欲讀入的資料夾名稱
# 讀入所有論文的文字資料，形成一個串列 Read all texts into a list.
papers = [] # 宣告空串列存放各篇論文文字
i = 0
for folder in folders:
    # 先產生各資料夾內所有檔案名稱的串列 Generate all filenames for each folder
    file_names = os.listdir(DATA_PATH + folder) # 'nipstxt' + 'nipsxx' 下的檔案名稱 'xxxx.txt'
    i = i + len(file_names) # 核驗總共有多少檔案 Verify how many files we have
    for file_name in file_names:
        with open(DATA_PATH + folder + '/' + file_name, encoding='utf-8', errors='ignore', mode='r+') as f: # f means file handler
            data = f.read() # 讀入該篇文字
        papers.append(data) # 將該篇文字添加於串列papers之後

# print('There are {} articles.\n'.format(i))
print('There are {} articles.\n'.format(len(papers)))

# 第一篇論文前1000個字 The first 1000 words in first article.
print(papers[0][:1000]) # 每篇論文的所有文字形成size為1的str，串接為串列papers的一個個元素

# a = ["I love you all.", "You love me too."]
# a[0][:6]

#### 基本文本整理 Basic Text Wrangling or Text (Pre)Processing
# !conda install nltk --y
import nltk

stop_words = nltk.corpus.stopwords.words('english') # 179個預設停用字 179 stopwords
# 正則表達式切分字詞
wtk = nltk.tokenize.RegexpTokenizer(r'\w+') # \w, \W: ANY ONE word/non-word character. For ASCII, word characters are [a-zA-Z0-9_] (https://www3.ntu.edu.sg/home/ehchua/programming/howto/Regexe.html)

print(wtk.tokenize('A bible in the church'))

# 詞形還原(lemmatization) versus 詞幹提取(stemming)
wnl = nltk.stem.wordnet.WordNetLemmatizer()

print(wnl.lemmatize('churches'))

def normalize_corpus(papers): # 語料庫清理工作 ~ 名詞有點混亂cleaning, wrangling, normalization, preprocessing...，而normalization似乎又有其他的涵意，eg. damping function and idf on tf
    norm_papers = [] # normalized tokens
    for paper in papers:
        paper = paper.lower() # 轉小寫 Transform to lower case
        paper_tokens = [token.strip() for token in wtk.tokenize(paper)] # .strip() 去掉前後多於的空白 Return a copy of the string with leading and trailing whitespace removed.
        paper_tokens = [wnl.lemmatize(token) for token in paper_tokens if not token.isnumeric()] # 詞形還原 Lemmatization
        paper_tokens = [token for token in paper_tokens if len(token) > 1] # 字元長度須大於 1 The length of token should be greater than 1
        paper_tokens = [token for token in paper_tokens if token not in stop_words] # 移除停用字詞 Remove stopwords
        paper_tokens = list(filter(None, paper_tokens)) # 移除空的元素 Remove empty token
        if paper_tokens: # bool([]) returns False, but bool(['anything inside']) returns True (https://stackoverflow.com/questions/10440792/why-does-false-evaluate-to-false-when-if-not-succeeds)
            norm_papers.append(paper_tokens)
            
    return norm_papers # normalized papers

# [token.strip() for token in wtk.tokenize(papers[0])]
# help('array'.strip) # Return a copy of the string S with leading and trailing whitespace removed.
# papers[0].lower()
# filter(None, ['paper_tokens', '']) # <filter at 0x7fc6b3fee650>
# 關於filter()方法, python3和python2有一點不同Python2.x 中傳回的是過濾後的列表, 而 Python3 中返回到是一個filter類別物件，此filter類別實現了__iter__ 和 __next__ 方法, 可以看成是一個迭代器, 有惰性運算的特性, 相對 Python2.x 提升了性能, 並可以節省記憶體空間。(https://www.runoob.com/python/python-func-filter.html)
# list(filter(None, ['paper_tokens', '']))

norm_papers = normalize_corpus(papers) # Length of 1740, same as 'papers'
print(len(norm_papers))
print(len(norm_papers[0])) # 第一篇論文的字數


#### 表達文本的屬性工程 Text Representation with Feature Engineering

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(min_df=20, max_df=0.6, ngram_range=(1,2), # max_df: corpus-specific stop words (特定於語料庫的停用詞) 此處文件頻率超出六成(0.6)者不計入詞彙中; min_df: this value is also called cut-off in the literature 此處表文件頻率低於20者不計入詞彙中 (https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
                     token_pattern=None, tokenizer=lambda doc: doc,
                     preprocessor=lambda doc: doc) # 傳入的資料物件已經分詞與前處理完成

cv_features = cv.fit_transform(norm_papers)
cv_features.shape # (1740->3480篇文件, 14408->14517詞項)

cv_features.todense()[:5, :50] # 確實稀疏！Seeing is believing.

vocabulary = np.array(cv.get_feature_names())
print('Total Vocabulary Size:', len(vocabulary))

print(type(cv.get_feature_names())) # <class 'list'>
print(len(cv.get_feature_names()))

### 法一：運用奇異值矩陣分解(PCA的計算方法之一)的潛在語義索引主題模型 Topic Models with Latent Semantic Indexing (LSI) by Singular Value Decomposition (SVD)
# PCA principal components analysis, SVD sigular value decomposition, LSI居然有關係！
from sklearn.decomposition import TruncatedSVD

TOTAL_TOPICS = 20 # 擷取20個主題 Extract twenty topics

lsi_model = TruncatedSVD(n_components=TOTAL_TOPICS, n_iter=500, random_state=42) # 勿忘統計機器學習建模語法 Don't forget the OO (object-oriented) syntax of statistical and machine learning
document_topics = lsi_model.fit_transform(cv_features) # (1740 docs, 20 topics) 文件 -> 主題的關係矩陣，可直接傳入稀疏矩陣(Compressed Sparse Row matrix, CSR)

### 檢視topics-terms矩陣
topic_terms = lsi_model.components_ # Relation between 20 topics and 14408 terms
topic_terms.shape # (20 topics, 14408 terms) 主題 -> 詞項

### numpy.argsort()解密
### https://numpy.org/doc/stable/reference/generated/numpy.argsort.html
x = np.array([3, 1, 2])
np.argsort(x)
x[np.argsort(x)] # 類似R語言中的 order 函數(values in, sorted indices out)，升冪排
np.argsort(-x)
x[np.argsort(-x)] # 降冪排

### 資料物件名釋義

# TOTAL_TOPICS: 擷取的主題數(20)
# top_terms: 各主題關鍵詞個數(20)
# topic_terms: 各主題與各詞項負荷loading/旋轉rotation矩陣(簡稱主題詞項負荷矩陣)
# topic_keyterm_idxs: 各主題前top_terms(20)關鍵詞索引
# topic_keyterm_weights: 各主題前top_terms(20)關鍵詞權重
# topic_keyterms: 各主題前top_terms(20)關鍵詞
# named_topic_keyterm_weights(注意物件名稱): 綑綁zip各主題下前20個相關的關鍵詞與其權重的串列

top_terms = 20
topic_keyterm_idxs = np.argsort(-np.absolute(topic_terms), axis=1)[:, :top_terms] # (20 topics, 20 top terms index) 主題詞項負荷矩陣取各主題下絕對值最大的前20個關鍵詞項 Returns the indices that would sort an array decreasingly.

topic_keyterm_weights = np.array([topic_terms[row, columns] 
                             for row, columns in list(zip(np.arange(TOTAL_TOPICS), topic_keyterm_idxs))]) # (主題編號, array([前20個正負相關詞項編號])) 再從主題詞項矩陣中取出關鍵詞項的權重 (20 topics, 20 top terms weights)

topic_keyterms = vocabulary[topic_keyterm_idxs] # 有趣的向量化語法 An interesting syntax ! (20 topics, 20 top terms)
named_topic_keyterm_weights = list(zip(topic_keyterms, topic_keyterm_weights)) # 物件名稱些許不同，而且資料結構亦不相同(把各主題下前20個相關的關鍵詞與其權重綑綁zip起來)

# Seeing is believing !
# named_topic_keyterm_weights[0]
# 列印結果外顯雙層for迴圈
for n in range(TOTAL_TOPICS):
    print('Topic #'+str(n+1)+':')
    print('='*50)
    d1 = [] # 正權重詞項(term, wt)
    d2 = [] # 負權重詞項(term, wt)
    terms, weights = named_topic_keyterm_weights[n] # try named_topic_keyterm_weights[0]
    term_weights = sorted([(t, w) for t, w in zip(terms, weights)], 
                          key=lambda row: -abs(row[1])) # 根據權重絕對值(abs)降冪(-)排序
    for term, wt in term_weights:
        if wt >= 0:
            d1.append((term, round(wt, 3))) # 小數點下三位
        else:
            d2.append((term, round(wt, 3)))
    # 印出正負權重關鍵詞(呈現方式與後兩方法LDA, NMF不同，因為此處是係數，後面是機率值)
    print('Direction 1:', d1)
    print('-'*50)
    print('Direction 2:', d2)
    print('-'*50)
    print()

### 再轉到docs-topics矩陣
# 1740篇論文對應到20個主題的座標值(分數矩陣 score matrix)
dt_df = pd.DataFrame(np.round(document_topics, 3), 
                     columns=['T'+str(i) for i in range(1, TOTAL_TOPICS+1)]) # (1740篇, 20主題)

dt_df.T

# 三篇論文最相關的三個主題
document_numbers = [13, 250, 500]

for document_number in document_numbers:
    top_topics = list(dt_df.columns[np.argsort(-np.absolute(dt_df.iloc[document_number].values))[:3]]) # Top 3
    print('Document #'+str(document_number)+':')
    print('Dominant Topics (top 3):', top_topics)
    print('Paper Summary:') # 該篇論文前500個字
    print(papers[document_number][:500])
    print()

# dt_df.iloc[13] # pandas Series
# dt_df.iloc[13].values # numpy ndarray


### 法二：運用潛在迪利克里特分配的主題模型 Topic Models with Latent Dirichlet Allocation (LDiA)
# Step 1: 載入類別函數
from sklearn.decomposition import LatentDirichletAllocation
TOTAL_TOPICS = 20 # 擷取20個主題 Extract twenty topics

# Step 2: 宣告空模
lda_model = LatentDirichletAllocation(n_components =TOTAL_TOPICS, max_iter=500, max_doc_update_iter=50,
                                      learning_method='online', batch_size=1740, learning_offset=50., 
                                      random_state=42, n_jobs=16)
# Step 3 & 4: 配適實模及轉換出文件主題矩陣
document_topics = lda_model.fit_transform(cv_features) # (1740 docs, 20 topics)

### 以下程式碼與SVD部份相似
### 檢視topics-terms矩陣
topic_terms = lda_model.components_ # (20 topics, 14408 terms)

top_terms = 20

topic_keyterm_idxs = np.argsort(-np.absolute(topic_terms), axis=1)[:, :top_terms] # (20 topics, top 20 terms index)

topic_keyterms = vocabulary[topic_keyterm_idxs] # 第四代程式語言特色 ~ 向量化 vectorization

topics = [', '.join(topic) for topic in topic_keyterms] # 'keyterm' and 'keyterms' seem more suitable. A list of length 20.

pd.set_option('display.max_colwidth', None)

topics_df = pd.DataFrame(topics,
                         columns = ['Terms per Topic'],
                         index=['Topic'+str(t) for t in range(1, TOTAL_TOPICS+1)]) # (1740 docs, 20 topics)

topics_df

### 再轉到docs-topics矩陣
# dt_df: docs-topics矩陣轉為pandas DataFrame

pd.options.display.float_format = '{:,.3f}'.format
dt_df = pd.DataFrame(document_topics, 
                     columns=['T'+str(i) for i in range(1, TOTAL_TOPICS+1)]) # (1740 docs, 20 topics)
dt_df.T # (20 topics, 1740 docs)

pd.options.display.float_format = '{:,.5f}'.format
pd.set_option('display.max_colwidth', 200)

# 各文件隸屬20各主題的機率總和為1
dt_df.sum(axis=1)

### 各個主題跨各文件最大機率值
# max_contrib_docs: 各主題貢獻度最高的論文機率值
# document_numbers: 各主題貢獻度最高的論文編號
# documents: 抓出文章成輯
# results_df: 各主題凌越論文機率值、論文索引、主題前20高關鍵字與論文內容

max_contrib_docs = dt_df.max(axis=0) # (20,) across docs (axis = 0) 
document_numbers = dt_df.idxmax(axis=0)

documents = [papers[i] for i in document_numbers] # 依論文篇號抓出文章成輯

results_df = pd.DataFrame({'Topic ID': max_contrib_docs.index, 'Contribution %': max_contrib_docs.values,
                          'Dominant Paper Num': document_numbers.values, 'Topic Keyterms': topics_df['Terms per Topic'].values, 
                          'Paper Content': documents})

results_df

### 各個文件跨各主題最大機率值
# max_contrib_topics: 各文件最可能談論主題的機率值
# dominant_topics_idx: 各文件最可能談論主題的索引值
# results_df2: 各文件凌越主題索引、機率值、論文編號與內容

max_contrib_topics = dt_df.max(axis=1) # (1740,) across topics (axis = 1)
dominant_topics_idx = dt_df.idxmax(axis=1) # (1740,)

results_df2 = pd.DataFrame({'Paper Num': range(1740), 'Dominant Topic': dominant_topics_idx, 'Contribution %': max_contrib_topics, 'Paper Content': papers})

results_df2

### 法三：運用非負矩陣分解的主題模型 Topic Models with Non-Negative Matrix Factorization (NMF)

from sklearn.decomposition import NMF

nmf_model = NMF(n_components=TOTAL_TOPICS, solver='cd', max_iter=1000, # Numerical solver to use: ‘cd’ is a Coordinate Descent solver. ‘mu’ is a Multiplicative Update solver.
                random_state=42, alpha=.1, l1_ratio=.85) # Constant that multiplies the regularization terms. Set it to zero to have no regularization. The regularization mixing parameter.

import time
start = time.time()
document_topics = nmf_model.fit_transform(cv_features) # ConvergenceWarning: Maximum number of iterations 500 reached. Increase it to improve convergence.
end = time.time()
print('Elapsed time is {} seconds.'.format(end - start))

### 以下程式碼與SVD部份相似，與LDiA相同
### 檢視topics-terms矩陣
topic_terms = nmf_model.components_ # (20 topics, 14408 terms)

topic_keyterm_idxs = np.argsort(-np.absolute(topic_terms), axis=1)[:, :top_terms] # (20 topics, top 20 terms index)

topic_keyterms = vocabulary[topic_keyterm_idxs]

topics = [', '.join(topic) for topic in topic_keyterms] # 'keyterm' and 'keyterms' seem more suitable. A list of length 20.

pd.set_option('display.max_colwidth', -1) # 'None' try it pls.

topics_df = pd.DataFrame(topics,
                         columns = ['Terms per Topic'],
                         index=['Topic'+str(t) for t in range(1, TOTAL_TOPICS+1)]) # (1740 docs, 20 topics)

topics_df

### 再轉到docs-topics矩陣
# dt_df: docs-topics矩陣轉為pandas DataFrame

pd.options.display.float_format = '{:,.3f}'.format

dt_df = pd.DataFrame(document_topics, 
                     columns=['T'+str(i) for i in range(1, TOTAL_TOPICS+1)]) # (1740 docs, 20 topics)
dt_df.head(10)


pd.options.display.float_format = '{:,.5f}'.format
pd.set_option('display.max_colwidth', 200)

# docs-topics矩陣內容並非機率值
dt_df.sum(axis=1)

### 各個主題跨各文件最大分數值
# max_score_docs: 各主題貢獻度最高的論文分數值
# document_numbers: 各主題貢獻度最高的論文編號
# documents: 抓出文章成輯
# results_df: 各主題凌越論文分數值、論文索引、主題前20高關鍵字與論文內容

max_score_docs = dt_df.max(axis=0) # (20,) across docs (axis = 0)

document_numbers = dt_df.idxmax(axis=0)

documents = [papers[i] for i in document_numbers] # 依論文篇號抓出文章成輯

results_df = pd.DataFrame({'Topic ID': max_score_docs.index, 'Contribution %': max_score_docs.values,
                          'Dominant Paper Num': document_numbers.values,
                           'Topic Keyterms': topics_df['Terms per Topic'].values, 
                          'Paper Content': documents})

results_df


### 各個文件跨各主題最大分數值
# max_contrib_topics: 各文件最可能談論主題的分數值
# dominant_topics_idx: 各文件最可能談論主題的索引值
# results_df2: 各文件凌越主題索引、分數值、論文內容

max_contrib_topics = dt_df.max(axis=1) # (1740,) across topics (axis = 1)
dominant_topics_idx = dt_df.idxmax(axis=1) # (1740,)

results_df2 = pd.DataFrame({'Paper Num': range(1740), 'Dominant Topic': dominant_topics_idx, 'Contribution %': max_contrib_topics, 'Paper Content': papers})

results_df2

### 預測新研究論文的主題 Predicting Topics for New Research Papers

import glob
# papers manually downloaded from NIPS 16
# https://papers.nips.cc/book/advances-in-neural-information-processing-systems-29-2016

new_paper_files = glob.glob('./test_data/nips16*.txt')

new_papers = []
for fn in new_paper_files:
    with open(fn, encoding='utf-8', errors='ignore', mode='r+') as f:
        data = f.read()
        new_papers.append(data)
              
print('Total New Papers:', len(new_papers))


norm_new_papers = normalize_corpus(new_papers)
cv_new_features = cv.transform(norm_new_papers)
cv_new_features.shape


topic_predictions = nmf_model.transform(cv_new_features) # (4 new papers, 20 topics)

best_topics = [[(topicID, round(score, 3)) 
                    for topicID, score in sorted(enumerate(topic_predictions[i]), 
                                            key=lambda row: -row[1])[:2]] 
                        for i in range(len(topic_predictions))]

best_topics # Best 2 topics

### sorted + enumerate 釋義
sorted(enumerate(topic_predictions[0]), key=lambda row: row[1]) # 升冪排列
sorted(enumerate(topic_predictions[0]), key=lambda row: -row[1]) # 升冪排列

results_df = pd.DataFrame()

results_df['Papers'] = range(1, len(new_papers)+1)

results_df['Dominant Topics'] = [[topic_num+1 for topic_num, sc in item] for item in best_topics]

res = results_df.set_index(['Papers'])['Dominant Topics'].apply(pd.Series).stack().reset_index(level=1, drop=True)

results_df = pd.DataFrame({'Dominant Topics': res.values}, index=res.index)

results_df['Topic Score'] = [topic_sc for topic_list in [[round(sc*100, 2) for topic_num, sc in item] for item in best_topics] for topic_sc in topic_list]

results_df['Topic Desc'] = [topics_df.iloc[t-1]['Terms per Topic'] for t in results_df['Dominant Topics'].values]

results_df['Paper Desc'] = [new_papers[i-1][:200] for i in results_df.index.values]

results_df


### 模型儲存和轉換器 Persisting Model and Transformers
### 可作為後續PyLDAViz使用
### This is just for visualizing the topics in the other notebook (since PyLDAViz expands the notebook size)

import dill

with open('nmf_model.pkl', 'wb') as f:
    dill.dump(nmf_model, f)
with open('cv_features.pkl', 'wb') as f:
    dill.dump(cv_features, f)
with open('cv.pkl', 'wb') as f:
    dill.dump(cv, f)

### Reference
# Sarkar, D. (2019), Text Analytics with Python, 2nd Edition, Apress.
