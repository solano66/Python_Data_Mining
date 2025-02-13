'''
Collated by Ching-Shih Tsou 鄒慶士 博士 (Ph.D.) Distinguished Prof. at the Department of Mechanical Engineering/Director at the Center of Artificial Intelligence & Data Science (機械工程系特聘教授兼人工智慧暨資料科學研究中心主任), MCUT (明志科技大學); Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), NTUB (國立臺北商業大學)
Notes: This code is provided without warranty.
'''


### 文字資料集群分析案例
import numpy as np
import pandas as pd
import nltk
import re # regular expression
import os

# load movie data
movie_data = pd.read_csv('../data/movie_data.csv')
# view movie data
print (movie_data.head()) # 教父、刺激1995、辛德勒的名單、蠻牛、卡薩布蘭加

titles = movie_data['Title']
synopses = movie_data['Synopsis']
print (titles[:10]) # first 10 titles 前十部電影名稱
print (synopses[0][:200]) # first 200 characters in first synopses (for 'The Godfather') 教父摘要前100個字

### 停用詞、詞幹化與符號化 Stopwords, stemming, and tokenizing
# load nltk's English stopwords as variable called 'stopwords'
stopwords = nltk.corpus.stopwords.words('english') # 179 stopwrods

print (stopwords[:10])

print (len(stopwords)) # 179

# load nltk's SnowballStemmer as variabled 'stemmer'
from nltk.stem.snowball import SnowballStemmer # There are other stemmers in nltk - Porter and Lancaster (演算法不同 http://www.nltk.org/api/nltk.stem.html?highlight=lemmatizer)
stemmer = SnowballStemmer("english")

# 查核dir()傳回的是屬性還是方法
[(name, type(getattr(stemmer, name))) for name in dir(stemmer)]
stemmer.stopwords # Empty set()

# here a tokenizer and stemmer are defined which returns the set of stems in the text that it is passed

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token 
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)] # 注意雙層嵌套式串列推導 embedded looping in a list comprehension
    # 過濾非字母標記 filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token): # 只留下字 only words are left (https://regex101.com)
           filtered_tokens.append(token)
    # 轉成詞幹
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text): # text is one synopsis of the corpus
# first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)] # 注意雙層嵌套式串列推導 embedded looping in a list comprehension
    # 過濾非字母標記 filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token): # 只留下字 only words are left
            filtered_tokens.append(token)
    return filtered_tokens

# 運用上面函數產生兩個詞彙集合 - 僅符號化的以及符號化後進一步完成詞幹化的詞彙集。
#not super pythonic, no, not at all.
#use extend so it's a big flat list of vocab
totalvocab_stemmed = [] # 簡單串列(flat list)存放分詞且詞幹化的所有詞彙
totalvocab_tokenized = [] # 簡單串列(flat list)存放分詞後的所有詞彙
for i in synopses:
    allwords_stemmed = tokenize_and_stem(i) # for each item in 'synopses', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed) # extend the 'totalvocab_stemmed' list

    allwords_tokenized = tokenize_only(i) # i是各篇電影摘要
    totalvocab_tokenized.extend(allwords_tokenized) # 154530

totalvocab_stemmed[:10]
len(totalvocab_stemmed)

totalvocab_tokenized[:10]
len(totalvocab_tokenized)

### 理解上面定義的函數 Understandand the funtion defined above.
def tokenize_and_stem_short(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    return tokens

synopses[0] # synopsis of the Godfather 因為輸出過長，請自行執行後檢視結果

tokens = tokenize_and_stem_short(synopses[0])

tokens[:25]

print (type(tokens))
print (tokens[1])
print (re.search('[a-zA-Z]', tokens[1]))

print (tokens[3])
print (re.search('[a-zA-Z]', tokens[3])) # None

filtered_tokens = []
# filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
for token in tokens:
    if re.search('[a-zA-Z]', token):
        filtered_tokens.append(token)
        
filtered_tokens[:25]

# 使用pandas模組的DataFrame資料結構呈現詞彙集
# index是字根的pandas字詞DataFrame
vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
print ('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')

print (vocab_frame.head(25))

### 詞頻-逆文件頻率與文件相似性
from sklearn.feature_extraction.text import TfidfVectorizer

# define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000, # max_df: corpus-specific stop words (特定於語料庫的停用詞) 此處文件頻率超出八成(0.8)者不計入詞彙中; min_df: this value is also called cut-off in the literature 此處表文件頻率低於兩成(0.2)者不計入詞彙中 (https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
                                   min_df=0.2, stop_words='english',
                                   use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

# %timeit tfidf.fit_transform(X_train)
# %time tfidf_matrix = tfidf_vectorizer.fit_transform(synopses) # fit the vectorizer to synopses
import time
start = time.time()
tfidf_matrix = tfidf_vectorizer.fit_transform(synopses) # fit the vectorizer to synopses
end = time.time()
print('Elapsed time: {}'.format(end - start))

print(tfidf_matrix.shape) # (100, 406)

### NLTK is a teaching toolkit. It's slow by design, because it's optimized for readability. (https://stackoverflow.com/questions/26195699/sklearn-how-to-speed-up-a-vectorizer-eg-tfidfvectorizer)
### Gensim has an efficient tf-idf model and does not need to have everything in memory at once. Your corpus simply needs to be an iterable, so it does not need to have the whole corpus in memory at a time. (https://stackoverflow.com/questions/25145552/tfidf-for-large-dataset)
dir(tfidf_vectorizer)
terms = tfidf_vectorizer.get_feature_names() # 406 terms
print(terms)

from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)
print(dist)
print(dist.shape) # (100, 100) a square matrix

### K-平均數集群
# 406個詞項維度不算高，歐幾里德直線距離結合k平均數法可行

from sklearn.cluster import KMeans

num_clusters = 5

km = KMeans(n_clusters=num_clusters)

# %time km.fit(tfidf_matrix)
import time
start = time.time()
km.fit(tfidf_matrix)
end = time.time()
print('Elapsed time: {}'.format(end - start))

dir(km)
clusters = km.labels_.tolist()

print(clusters)

from sklearn.externals import joblib # https://newbedev.com/what-are-the-different-use-cases-of-joblib-versus-pickle

# uncomment the below to save your model
# since I've already run my model I am loading from the pickle

# joblib.dump(km, 'doc_cluster.pkl')

# km = joblib.load('doc_cluster.pkl')
# clusters = km.labels_.tolist()
# print(clusters)

films = { 'title': titles, 'synopsis': synopses, 'cluster': clusters}
frame = pd.DataFrame(films, index = range(100))
# frame = pd.DataFrame(films, index = [clusters] , columns = ['rank', 'title', 'cluster', 'genre'])

frame

# 各群中影片數
frame['cluster'].value_counts() # number of films per cluster (clusters from 0 to 4)

# 了解各群中的電影主題
# from __future__ import print_function # 在python2的環境下，超前使用python3的print函數，我們的情境不需要 (https://zhuanlan.zhihu.com/p/28641474)

# print("Top terms per cluster:")
# print()
# sort cluster centers by proximity to centroid
# order_centroids = km.cluster_centers_.argsort()[:, ::-1]

# for i in range(num_clusters):
#     print("Cluster %d words:" % i, end='') # Python’s print() function comes with a parameter called ‘end’. By default, the value of this parameter is ‘\n’, i.e. the new line character. You can end a print statement with any character/string using this parameter. (https://www.geeksforgeeks.org/gfact-50-python-end-parameter-in-print/)

#     for ind in order_centroids[i, :6]: # replace 6 with n words per cluster
#         print(' %s' % vocab_frame.loc[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
#         print() # add whitespace
#         print() # add whitespace

#         print("Cluster %d titles:" % i, end='')
#         for title in frame.loc[i, 'title']: # 原 for title in frame.ix[i]['title'].values.tolist()
#             print(' %s,' % title, end='')
#         print() # add whitespace
#         print() # add whitespace

# print()
# print()

### (補充)numpy ndarray 的 argsort + ::-1 釋義
np.arange(6).reshape((2,3))
np.arange(6).reshape((2,3)).argsort(axis=1)
np.arange(6).reshape((2,3)).argsort()[:, ::-1]
##############################################

km.cluster_centers_

order_centroids = km.cluster_centers_.argsort()[:, ::-1]

order_centroids

cen = pd.DataFrame(km.cluster_centers_, index = range(5), columns = terms)

cen_20terms = cen.columns.values[order_centroids][:, :20]
print(cen_20terms)

cen_20terms_weights = np.array([km.cluster_centers_[row, columns] 
                             for row, columns in list(zip(range(cen.shape[0]), order_centroids[:, :20]))]) # (群編號, array([前20個高中心詞項編號])) 再從群中心座標矩陣中取出各群關鍵詞項的權重 (5 clusters, 20 top terms weights)
print(cen_20terms_weights)

### 多維尺度法
import os # for os.path.basename

import matplotlib.pyplot as plt
# import matplotlib as mpl

from sklearn.manifold import MDS

MDS()

# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

pos = mds.fit_transform(dist) # shape (n_samples, n_components), attention to the distance matrix, not the tfidf matrix !

xs, ys = pos[:, 0], pos[:, 1]

### 視覺化文件集群
# set up colors per clusters using a dict
cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3:'#e7298a', 4: '#66a61e'}

# set up cluster names using a dict
cluster_names = dict(pd.DataFrame(cen_20terms).apply(lambda x: ','.join(x[:4]), axis=1))
# cluster_names = {0: 'Family, home, war',
#                  1: 'Police, killed, murders',
#                  2: 'Father, New York, brothers',
#                  3: 'Dance, singing, love',
#                  4: 'Killed, soldiers, captain'}

# some ipython magic to show the matplotlib plots inline
#%matplotlib inline

# create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles)) # (x座標, y座標, 集群編號, 片名)

# group by cluster
groups = df.groupby('label')

# set up plot
fig, ax = plt.subplots(figsize=(17, 9)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

# iterate through groups to layer the plot
# note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
            label=cluster_names[name], color=cluster_colors[name],
            mec='none')
ax.set_aspect('auto')
ax.tick_params(\
   axis= 'x', # changes apply to the x-axis
   which='both', # both major and minor ticks are affected
   bottom='off', # ticks along the bottom edge are off
   top='off', # ticks along the top edge are off
   labelbottom='off')

ax.tick_params(\
   axis= 'y', # changes apply to the y-axis
   which='both', # both major and minor ticks are affected
   left='off', # ticks along the bottom edge are off
   top='off', # ticks along the top edge are off
   labelleft='off')

ax.legend(numpoints=1) # show legend with only 1 point

# add label in x, y position with the label as the film title
for i in range(len(df)):
    ax.text(df.iloc[i]['x'], df.iloc[i]['y'], df.iloc[i]['title'], size=8)


plt.show() # show the plot


# uncomment the below to save the plot if need be
# plt.savefig('clusters_small_noaxes.png', dpi=200)

plt.close()


### Reference:
# https://github.com/brandomr/document_cluster