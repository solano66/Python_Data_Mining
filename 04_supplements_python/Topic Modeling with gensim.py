'''
Collated by Ching-Shih Tsou 鄒慶士 博士 (Ph.D.) Distinguished Prof. at the Department of Mechanical Engineering/Director at the Center of Artificial Intelligence & Data Science (機械工程系特聘教授兼人工智慧暨資料科學研究中心主任), MCUT (明志科技大學); Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), NTUB (國立臺北商業大學)
Notes: This code is provided without warranty.
'''

### Topic Modeling with Gensim (Python)
### 1. 匯入套件 Import Packages
# Run in python console
import nltk
# nltk.download('stopwords') # only needed at the first time

# Run in terminal or command prompt
# python3 -m spacy download en

import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
# conda install -c anaconda gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
# %matplotlib inline

# Enable logging for gensim - optional
# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

# import warnings
# warnings.filterwarnings("ignore",category=DeprecationWarning)

### The following are key factors to obtaining good segregation topics:

# The quality of text processing.
# The variety of topics the text talks about.
# The choice of topic modeling algorithm.
# The number of topics fed to the algorithm.
# The algorithms tuning parameters.

### 2. 準備停用字詞 Prepare Stopwords
# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

list(wd for wd in ["a", "f"] if wd in ["a", "b", "c", "d", "e"])
list(wd for wd in ['from', 'subject', 're', 'edu', 'use'] if wd in stop_words)

stop_words.extend(['subject', 'edu', 'use'])

### 3. 匯入新聞群組資料 Import Newsgroups Data
# Import Dataset
df = pd.read_json('https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json')

df.dtypes
print(df.target_names.unique())
len(df.target_names.unique()) # 20
df.target.value_counts()
len(df.target.value_counts()) # 20

df.head(15)

### 4. 移除電子郵件與換行字元 Remove emails and newline characters (Please check the metacharacter part in RegularExpression.py)
# Convert to list
data = df.content.values.tolist()

# Remove Emails
data[3]

data = [re.sub(r'\S*@\S*\s?', '', sent) for sent in data] # Try \w*@\w*\s? and \w*@\S*\s? at https://regex101.com on data[3]

# Remove new line characters
data = [re.sub(r'\s+', ' ', sent) for sent in data] # 多個空格替換成一個空格

# Remove distracting single quotes
data = [re.sub("\'", "", sent) for sent in data]

pprint(data[:1])

data[3]

### 5. 切詞與清理 Tokenize words and Clean-up text
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(data)) # generator to list

print(data_words[:1])

### 6. Creating Bigram and Trigram Models
# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
print(trigram_mod[bigram_mod[data_words[2]]])

### 7. 移除停用詞、生成二元詞與進行詞幹化 Remove Stopwords, Make Bigrams and Lemmatize
# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en', disable=['parser', 'ner']) # Please refer to Language Processing Pipelines (https://spacy.io/usage/processing-pipelines)

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized[:1])

### 8. 建立字典與語料庫 Create the Dictionary and Corpus needed for Topic Modeling
# Create Dictionary (先建構字典)
id2word = corpora.Dictionary(data_lemmatized)
dir(id2word)
len(list(id2word.items())) # 39838

# Create Corpus (再建語料庫)
texts = data_lemmatized

# Term Document Frequency (運用doc2bow)
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:2])

### corpora.Dictionary 釋義 (https://radimrehurek.com/gensim/corpora/dictionary.html)
texts = [['human', 'interface', 'computer']] # [(0, 'computer'), (1, 'human'), (2, 'interface')]
dct = corpora.Dictionary(texts)  # initialize a Dictionary
list(dct)
dct.add_documents([["cat", "say", "meow"], ["dog"]])  # add more document (extend the vocabulary), [(3, "cat"), (4, "meow"), (5, "say"), (6, "dog")]
list(dct) # four more words
dct.doc2bow(["dog", "computer", "non_existent_word"])
#######################

id2word[0] # 'addition'

# Human readable format of corpus (term-frequency)
[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]] # 值得細細品味的雙層巢狀迴圈


### 9. 建立主題模型 Building the Topic Model
# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=20, random_state=100, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)

# alpha and eta are hyperparameters that affect sparsity of the topics. According to the Gensim docs, both defaults to 1.0/num_topics prior.
# chunksize is the number of documents to be used in each training chunk.
# update_every determines how often the model parameters should be updated and
# passes is the total number of training passes.

### 10. 檢視主題關鍵字 View the topics in LDA model
# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

# Topic 0 is a represented as 0.175*"file" + 0.073*"entry" + 0.057*"error" + 0.053*"display" +  0.040*"program" + 0.030*"sun" + 0.025*"version" + 0.024*"cool" + 0.020*"output" + 0.020*"crash".
# It means the top 10 keywords that contribute to this topic are: "file", "entry", "error".. and so on and the weight of "file" on topic 0 is 0.175.

### 11. 計算模型困惑度與一致性分數 Compute Model Perplexity and Coherence Score
# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

### 主題-關鍵字視覺化 Visualize the topics-keywords
# Visualize the topics
# pyLDAvis.enable_notebook()
# vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
# vis

### 12. 決定最佳主題數 How to find the optimal number of topics for LDA?
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics, random_state=100, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

# Can take a long time to run.
model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=2, limit=40, step=6) # (40 - 2)/6 + 1 ~= 7 different # of topics

from sklearn.externals import joblib

lda_tuning = 'lda_tuning.pkl'
# joblib.dump([model_list, coherence_values], lda_tuning)

model_list_restored, coherence_values_restored = joblib.load(lda_tuning)

# Show graph
limit=40; start=2; step=6;
x = range(start, limit, step)
plt.plot(x, coherence_values_restored)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

# Print the coherence scores
for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
    
# Select the model and print the topics
optimal_model = model_list[3]
model_topics = optimal_model.show_topics(formatted=False)
pprint(optimal_model.print_topics(num_words=10))

### Reference:
# Topic Modeling with Gensim (https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/)
