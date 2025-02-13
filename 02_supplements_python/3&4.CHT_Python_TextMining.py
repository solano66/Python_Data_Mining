'''
Collated by Ching-Shih Tsou 鄒慶士 博士 (Ph.D.) Distinguished Prof. at the Department of Mechanical Engineering/Director at the Center of Artificial Intelligence & Data Science (機械工程系特聘教授兼人工智慧暨資料科學研究中心主任), MCUT (明志科技大學); Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), NTUB (國立臺北商業大學); the Chinese Academy of R Software (CARS) (中華R軟體學會創會理事長) and the Data Science and Business Applications Association of Taiwan (DSBA) (臺灣資料科學與商業應用協會創會理事長); Standing Supervisor at the Chinese Association of Quality Assessment and Evaluation (CAQAE) (社團法人中華品質評鑑協會常務監事); Chairman of the Committee of Big Data Quality Applications at the Chinese Society of Quality (CSQ) (社團法人中華民國品質學會大數據品質應用委員會主任委員)
Notes: This code is provided without warranty.
'''

### (補充) 字串文字基礎 Strings Fundamentals
### A. 簡單字串 Simple string
a = 'this is a string'
# 檢視字串的類別
type(a)

# 字串的串接運算子'+'
a = 'this is the first half '
b = 'and this is the second half'
a + b

simple_string = 'hello' + " I'm a simple string"
simple_string

### 取值(用[])與不可更動 Subsetting and immutable
print(a[10])
# a[10] = 'f' # TypeError: 'str' object does not support item assignment

### 型別轉換 Type conversion
a = 5.6
type(a)
s = str(a)
s
type(s)

### B. 多列字串 Multi-line string, note the \n (newline) escape character automatically created
multi_line_string = """Hello I'm
a multi-line
string!"""
multi_line_string # 結果會出現逃脫(escape)字元
# Attention to the difference to above
print(multi_line_string) # 注意與無print()的差異！without Out [xx] 且會對逃脫字元進行釋義(ex. '\n'表示換列)

### C. 字串中的逃脫字序 Escape sequences in strings
### 逃脫字序的起始字符是反斜線'\'，後接ASCII字符，例如：換列符號'\n'、定位符號'\t'。These strings often have escape sequences embedded in them, where the rule for escape sequences starts with a backslash (\) followed by any ASCII character. Hence, they perform backspace interpolation. Popular escape sequences include (\n), which indicates a newline character and (\t), indicating a tab.

### D. 原始字串 Raw Strings
### Windows的路徑分隔符號
### 路徑字串與逃脫字串 Normal string with escape sequences leading to a wrong file path!
escaped_string = "C:\the_folder\new_dir\file.txt"
print(escaped_string) # will cause errors if we try to open a file, because \t, \n, and \f here (勿忘前面所提到的逃脫字元釋義)
# type(escaped_string) # str

# raw string keeping the backslashes in its normal form (字串前加修飾字 r，使得逃脫字元不被釋義)
raw_string = r'C:\the_folder\new_dir\file.txt'
print(raw_string)
# type(escaped_string) # str

### E. 萬國碼與位元碼字串文字 Unicode and bytecode string literals
# https://unicode-table.com/cn/2639/
smiley = u"\u263A"

print(smiley) # '☺'

type(smiley) # str

# (u"\u263A").decode()
# AttributeError: 'str' object has no attribute 'decode' ('str'物件沒有解碼方法，'bytes'物件才有！)

ord(smiley) # = 9786 decimal value of code point = 263A in Hex. (Return the Unicode code point for a ***one-character*** string. ord()函數傳回**單一**字元萬國碼十進位編碼點)

len(smiley) # 1

smiley.encode('utf8') # prints '\xe2\x98\xba' the bytes - it is <str>

type(smiley.encode('utf8')) # 笑臉符號(萬國碼)被編碼為位元組碼(bytes)

print (b'\xe2\x98\xba') # b'\xe2\x98\xba'

(b'\xe2\x98\xba').decode() # A smiling face

len(smiley.encode('utf8')) # its length = 3, because it is encoded as 3 bytes (長度為3，因為笑臉符號被編碼為三個位元組)

# print(u"\u263A".encode('ascii')) # 'ascii' codec can't encode character '\u263a' in position 0: ordinal not in range(128)
 

### 字串文字小結 String Literals Conclusions (再多看一些例子)
# unicode string literals
string_with_unicode = u'H\u00e8llo!'
print(string_with_unicode)
ord(string_with_unicode)
# TypeError: ord() expected a character, but string of length 5 found (為何錯誤？因為ord()只能對**單一**字元吐出其萬國碼十進位編碼點)
type(string_with_unicode) # str

string_with_unicode = u'H\xe8llo'
print(string_with_unicode)
ord(string_with_unicode)
# TypeError: ord() expected a character, but string of length 5 found (為何錯誤？因為ord()只能對**單一**字元吐出其萬國碼十進位編碼點)
type(string_with_unicode) # str

# What is the difference between using \u and \x while representing character literal?
# https://stackoverflow.com/questions/32175482/what-is-the-difference-between-using-u-and-x-while-representing-character-lite
# \x consumes 1-4 characters (\x hex-digit hex-digitopt hex-digitopt hex-digitopt), so long as they're hex digits - whereas \u must always be followed by 4 hex digits.
# I would strongly recommend only using \u, as it's much less error-prone.

u'H\u00e8llo!'.encode('utf8') # b'H\xc3\xa8llo!'
u'H\u00e8llo!'.encode() # b'H\xc3\xa8llo!'

u'H\xe8llo'.encode('utf8') # b'H\xc3\xa8llo'
u'H\xe8llo'.encode() # b'H\xc3\xa8llo'，預設設定值正是'utf8'

# '\u00A9' == '\x00A9' # False
# u'H\ue8llo!' == u'H\xe8llo'
# 'H\ue8llo!' == 'H\xe8llo'

### 字串格式化 String formatting
template = '%.2f %s are worth $%d'
template % (31.5560, 'Taiwan Dollars', 1)

# %d 整數
# %f 浮點數
# %.2f 浮點數,小數點 2 位
'We have %d %s containing %.2f gallons of %s' %(2, 'bottles', 2.5, 'milk')

# 注意會有進位
'We have %d %s containing %.2f gallons of %s' %(5.21, 'jugs', 10.86763, 'juice')

# 新式寫法
'Hello {} {}, it is a great {} to meet you at {}'.format('Mr.', 'Jones', 'pleasure', 5)

'I have a {food_item} and a {drink_item} with me'.format(drink_item='soda', food_item='sandwich')

print('{0:.2f} {1} are worth ${2}'.format(31.5560, 'Taiwan Dollars', 1))

### 英文切分工具(首次使用要下載數據 nltk.download())
### 分句器 Sentence tokenizers
### A. nltk中的sent_tokenize類別函數
import nltk # Natural Language ToolKit
from pprint import pprint

from nltk.corpus import gutenberg # The Project Gutenberg electronic text archive contains some 25,000 free electronic books, hosted at http://www.gutenberg.org/.
nltk.corpus.gutenberg.fileids()

# average word length, average sentence length, and the number of times each vocabulary item appears in the text on average (our lexical diversity score)
for fileid in gutenberg.fileids():
    num_chars = len(gutenberg.raw(fileid))
    num_words = len(gutenberg.words(fileid))
    num_sents = len(gutenberg.sents(fileid))
    num_vocab = len(set(w.lower() for w in gutenberg.words(fileid)))
    print(round(num_chars/num_words), round(num_words/num_sents), round(num_words/num_vocab), fileid)

# nltk.set_proxy("http://proxy.cht.com.tw:8080")
# ntlk.download()
# 第一次執行需要下載語料庫
# nltk.download('gutenberg')
alice = gutenberg.raw(fileids='carroll-alice.txt') # 長文章
sample_text = 'We will discuss briefly about the basic syntax, structure and design philosophies. There is a defined hierarchical syntax for Python code which you should remember when writing code! Python is a really powerful programming language!' # 短句
# 總字元數 Total characters in Alice in Wonderland
print (len(alice))

# 前300個字元 First 300 characters in the corpus
print (alice[0:300])
dir(nltk)
# 定義分句器
default_st = nltk.sent_tokenize
alice_sentences = default_st(text = alice)
sample_sentences = default_st(text = sample_text)

print ('\nTotal sentences in alice:', len(alice_sentences))
print ('First 5 sentences in alice:-')
pprint(alice_sentences[0:5])
print ('Total sentences in sample_text:', len(sample_sentences))
print ('Sample text sentences :-')
pprint(sample_sentences) # Please compare to print(sample_sentences)

### B. PunktSentenceTokenizer
punkt_st = nltk.tokenize.PunktSentenceTokenizer()

alice_sentences = punkt_st.tokenize(alice)
pprint(alice_sentences)
sample_sentences = punkt_st.tokenize(sample_text)
pprint(sample_sentences)

### 分詞器 Word tokenizers

### A. word_tokenize類別函數
sentence = "The brown fox wasn't that quick and he couldn't win the race"
# 定義分詞器
default_wt = nltk.word_tokenize
words = default_wt(sentence)
print (words)

### B. TreebankWordTokenizer
treebank_wt = nltk.TreebankWordTokenizer()
words = treebank_wt.tokenize(sentence)
print (words)

### 整合運用：文本符號化 Tokenizing Text
### 自定義函數'tokenize_text'先分句，接著分詞
import nltk
import re
import string
from pprint import pprint
corpus = ["The brown fox wasn't that quick and he couldn't win the race",
"Hey that's a great deal! I just bought a phone for $199",
"@@You'll (learn) a **lot** in the book. Python is an amazing language!@@"]

def tokenize_text(text):
    sentences = nltk.sent_tokenize(text) # 分句
    word_tokens = [nltk.word_tokenize(sentence) for sentence in sentences] # 分詞
    return word_tokens

token_list = [tokenize_text(text) for text in corpus]
pprint(token_list)

### 中文分句(http://blog.csdn.net/laoyaotask/article/details/9260263)
# 手動分句
# 設置分句的標點符號；可以根據實際需要進行修改。
cutlist = "。！？"

# 檢查某字符是否分句標點符號的函數；如果是，返回True，否則返回False。
def FindToken(cutlist, char):  
    if char in cutlist:  
        return True  
    else:  
        return False

print(FindToken(cutlist, ':'))
print(FindToken(cutlist, ';'))
print(FindToken(cutlist, '!')) # 半形
print(FindToken(cutlist, '！')) # 全形

def Cut(cutlist, lines): #參數1：設定分句標點符號；參數2：被分句的文本，為一列中文字符  
    l = []         #句子串列，用於儲存分句成功後的內容，為函數的回傳值
    line = []    #臨時串列，用於儲存捕獲到分句標點符之前的每個字符，一旦發現分句符號後，就會將其內容全部賦給l，然後就會被清空  
          
    for i in lines:         #對函數參數2中的每一字符逐個進行檢查  
        if FindToken(cutlist,i):       #如果當前字符是分句符號  
            line.append(i)          #將此字符放入臨時串列中  
            l.append(''.join(line))   #並把當前臨時串列的內容加入到句子串列中  
            line = []  #將符號串列清空，以便下次分句使用  
        else:         #如果當前字符不是分句符號，則將該字符直接放入臨時串列中  
            line.append(i)       
    return l


doc = "資料處理與分析是當代IT顯學，在大數據當道的時代，讓我們來探討搜尋引擎與文字探勘的應用。文字探勘，也被稱為文本挖掘、文字採礦、智慧型文字分析、文字資料探勘或文字知識發現，一般而言，指的是從非結構化的文字中，萃取出有用的重要資訊或知識。資料探勘(Data Mining)與文字探勘(Text Mining)關係緊密，相較於前者顯著的結構化，後者長短不一、沒有規律，且尚有現今生活中隨口說出來的慣用語。與Data Mining不同之處，在於Text Mining是針對文字進行分析，且低結構化的文字居多。TF-IDF是一種用於資訊檢索與文字探勘的常用加權技術，為一種統計詞頻的方法。"

l = Cut(list(cutlist), list(doc)) # 標點符號設定為"。！？"
l

### 中文分詞工具
### 三種分詞模式
import jieba # !conda install -c conda-forge jieba --y
seg_list = jieba.cut("我訪問南京理工大學與位於蘇州的西交利物浦大學", cut_all=True)
# Full Mode: 我/ 訪/ 問/ 南京/ 南京理工/ 理工/ 理工大/ 工大/ 學/ 與/ 位/ 於/ 蘇/ 州/ 的/ 西/ 交/ 利物/ 利物浦/ 大/ 學
seg_list = jieba.cut("我访问南京理工大学与位于苏州的西交利物浦大学", cut_all=True)
print("Full Mode: " + "/ ".join(seg_list)) #全模式
# Full Mode: 我/ 访问/ 南京/ 南京理工/ 理工/ 理工大/ 理工大学/ 工大/ 大学/ 与/ 位于/ 苏州/ 的/ 西/ 交/ 利物/ 利物浦/ 大学

seg_list = jieba.cut("我訪問南京理工大學與位於蘇州的西交利物浦大學", cut_all=False) # 位/ 於
seg_list = jieba.cut("我访问南京理工大学与位于苏州的西交利物浦大学", cut_all=False) # 位於
print("Default Mode: " + "/ ".join(seg_list)) #精確模式

seg_list = jieba.cut("他來到网易杭研大厦") #預設為精確模式
print(", ".join(seg_list))

seg_list = jieba.cut_for_search("小明碩士畢業於中國科學院計算所，後在日本京都大學深造") #搜索引擎模式
print(", ".join(seg_list))

sent="現在全台的電力供應已經漸漸恢復正常。對於這次停電，我代表政府向全國人民道歉。供電不只是民生問題，它還是國家安全的問題。真正該全面檢討的，是這個會因為人為疏失而輕易癱瘓的供電系統。這才是問題的核心。所以，我會要求相關部門，在最短時間內給全體國人一個清楚的報告，為什麼我們的供電系統會因為一個誤觸，就造成這麼大的損害？這個系統明顯地過於脆弱，可是台灣竟然就這樣過了這麽多年。這個系統非改不可，我會把它列為未來徹底檢討改革的重點。現在政府推動分散式的綠能發電，就是要避免單一電廠事故就影響全國供電。我們的政策方向不會改變，今天的事件只會讓我們的決心更堅定。"

print ("Input ", sent)

#ret = open("speech.txt", "r").read()
seglist = jieba.cut(sent, cut_all=False)
seglist # A generator (吃不到的物件！)

"/".join(seglist)

seglist = jieba.lcut(sent, cut_all=False)
seglist[:14]

### 自定義停用詞
stopwords = {}.fromkeys(['的', '附近'])

stopwords

list(stopwords)

### 載入停用詞檔案
import os
os.chdir("/Users/Vince/Google Drive/TextMining/III_scripts/data")
stopwords = {}.fromkeys([line.rstrip() for line in open('stopWords_Trad.txt') ]) # encoding='utf-8' for Windows
list(stopwords)[:20] # same as list(stopwords.keys())[:20]

type(stopwords.keys())

'乘機' not in stopwords

'停電' not in stopwords

### 將分詞結果刪除停用詞
def remove_stopwords(tokens):
    stopword_list = list(stopwords)
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    return filtered_tokens

filtered_list = remove_stopwords(seglist)
print (filtered_list)

"/".join(seglist)

### nltk中的英文停用詞
len(nltk.corpus.stopwords.words('english'))
"/".join(nltk.corpus.stopwords.words('english'))

def remove_stopwords(tokens):
    stopword_list = nltk.corpus.stopwords.words('english')
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    return filtered_tokens

corpus = ["The brown fox wasn't that quick and he couldn't win the race",
"Hey that's a great deal! I just bought a phone for $199",
"You'll (learn) a lot in the book. Python is an amazing language!"]

corpus_tokens = [tokenize_text(text) # 前面定義的分句分詞函數tokenize_text()
                          for text in corpus]

filtered_list = [[remove_stopwords(tokens)
                       for tokens in sentence_tokens]
                       for sentence_tokens in corpus_tokens]

print (filtered_list)

### 新增新詞到詞典jieba.add_word()和jieba.load_userdict()
print ("/".join(jieba.cut("江州市长江大桥参加了长江大橋的通车仪式")))

jieba.add_word("江大桥",freq =50000)
#jieba.add_word("江大桥")

print ("/".join(jieba.cut("江州市长江大桥参加了长江大橋的通车仪式")))

test_sent = (
"李小福是創新辦主任也是雲計算方面的專家：什麼是八一雙鹿\n"
"例如我輸入一個帶韓玉賞鑒的標題，在自定義詞庫中也增加了此詞為N類\n"
"「台中」正確應該不會被切開。mac上可分出「石墨烯」：此時又可以分出來凱特琳了。"
)

words = jieba.cut(test_sent)
print('/'.join(words))

jieba.load_userdict("./data/userdict.txt")

words = jieba.cut(test_sent)
print('/'.join(words))

### 調整詞典
print('/'.join(jieba.cut('如果放到post中將出錯。', HMM=False)))

jieba.suggest_freq(('中將'), True)

print('/'.join(jieba.cut('如果放到post中將出錯。', HMM=False)))

### 中文詞性標註(https://gist.github.com/luw2007/6016931)
import jieba.posseg as pseg
# 還是一個generator
words = pseg.cut("中文分詞於中文文字處理中視極為重要的一個環節，而中文文字本身就屬於一個開放集合，不存在一個完整的字典就可以列出所有的詞彙，在處理不同領域的文件時，會出現該相關領域的專有名詞，常常會有詞彙不足而導致切分錯誤的問題。")
for word, flag in words:
    print('%s %s' % (word, flag))

### 有時要關掉HMM
'/'.join(jieba.cut('丰田太省了', HMM=True))

'/'.join(jieba.cut('丰田太省了', HMM=False))

'/'.join(jieba.cut('我們中出了一个叛徒', HMM=True))

'/'.join(jieba.cut('我們中出了一个叛徒', HMM=False))

### 關鍵詞
from jieba import analyse
# 引入TF-IDF關鍵詞抽取接口
tfidf = analyse.extract_tags

# 原始語句
text = "線程市程序執行時的最小單位，它是進程的一個執行流，\
        是CPU調度和分派的基本單位，一個進程可以由很多個線程組成，\
        線程間共享進程的所有資源，每個線程有自己的堆棧和局部變量。\
        線程由CPU獨立調度執行，在多CPU環境下就允許多個線程同時運行。\
        同樣多線程也可以實現並發操作，每個請求分配一個線程來處理。"
        
# 基於TF-IDF算法進行關鍵詞抽取
keywords = tfidf(text, withWeight=True)
print ("keywords by tfidf:")

# 輸出抽取出的關鍵詞
for keyword in keywords:
    print (keyword[0] + "/",)

from jieba import analyse
analyse.textrank(text, topK=20)

### 文字雲
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import jieba
import codecs
text_from_file_with_apath = codecs.open('/Users/Vince/cstsouMac/Python/Examples/TextMining/3KDText_Mac_short.txt', encoding='utf-8').read()

wordlist_after_jieba = jieba.cut(text_from_file_with_apath, cut_all = False)
wl_space_split = " ".join(wordlist_after_jieba)
wl_space_split

my_wordcloud = WordCloud(font_path = './data/msyh.ttf').generate(wl_space_split)
plt.imshow(my_wordcloud)
plt.axis("off")
plt.show()

import matplotlib.pyplot as plt
import pickle
from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator
import jieba
import codecs
fin = codecs.open('/Users/Vince/cstsouMac/Python/Examples/TextMining/3KDText_Mac_short.txt', mode = 'r', encoding = 'utf-8')
print (fin.read())

# 第一次執行程序時將分好的詞存入文件
text = ''
with open('/Users/Vince/cstsouMac/Python/Examples/TextMining/3KDText_Mac_short.txt') as fin:
     for line in fin.readlines():
         line = line.strip('\n')
         text += ' '.join(jieba.cut(line))
         text += ' '
fout =
open('/Users/Vince/cstsouMac/Python/Examples/TextMining/text.txt','wb')
pickle.dump(text,fout)
fout.close()

# 直接從文件讀取數據
fr =
open('/Users/Vince/cstsouMac/Python/Examples/TextMining/text.txt','rb')
text = pickle.load(fr)

backgroud_Image = plt.imread('/Users/Vince/cstsouMac/Python/Examples/TextMining/girl.jpg')
wc = WordCloud( background_color = 'white', # 設定背景顏色
                mask = backgroud_Image, # 設定背景圖片
                max_words = 2000, # 設定最大數量的字數
                stopwords = STOPWORDS, # 設定停用詞
                font_path = '/Users/Vince/cstsouMac/Python/Examples/TextMining/msyh.ttf', # 設定字體格式，如不設定顯示不了中文
                max_font_size = 50, # 設定字體最大值
                random_state = 30, # 設定有多少種隨機生成狀態，即有多少種配色方案
)
wc.generate(text)
image_colors = ImageColorGenerator(backgroud_Image)
wc.recolor(color_func = image_colors)
plt.imshow(wc)
plt.axis('off')
plt.show()

### 詞袋模型與文件詞項矩陣
### 方法1 手動使用 counter
from collections import Counter
from itertools import chain
import numpy as np

documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]

def word_matrix(documents):
    '''計算詞頻矩陣(詞項-文件矩陣)'''
    # 所有字母轉換位小寫
    docs = [d.lower() for d in documents]
    
    # 分詞
    docs = [d.split() for d in docs]
    
    # 獲取所有詞
    words = list(set(chain(*docs)))
    
    #print(words)
    # 詞到ID的對映, 使得每個詞有一個ID
    dictionary = dict(zip(words, range(len(words))))
    #print(dictionary)
    
    # 建立一個空的矩陣, 行數等於詞數, 列數等於文件數
    matrix = np.zeros((len(words), len(docs)))
    
    # 逐個文件統計詞頻
    for col, d in enumerate(docs):  # col 表示矩陣第幾列，d表示第幾個文件。
        # 統計詞頻
        count = Counter(d)#其實是個詞典，詞典元素為：{單詞：次數}。
        for word in count:
            # 用word的id表示word在矩陣中的行數，該文件表示列數。
            id = dictionary[word]
            # 把詞頻賦值給矩陣
            matrix[id, col] = count[word]
    return matrix, dictionary

matrix, dictionary = word_matrix(documents)

print(matrix,'\n',dictionary)

### 方法2 使用 sklearn.feature_extraction.text.CountVectorizer
CORPUS = [
'the sky is blue',
'sky is blue and sky is beautiful',
'the beautiful sky is so blue',
'i love blue cheese'
]

new_doc = ['loving this blue sky today']

import sklearn
from sklearn.feature_extraction.text import CountVectorizer

dir(sklearn.feature_extraction.text) # 有'CountVectorizer', 'HashingVectorizer', 'TfidfVectorizer'

def bow_extractor(corpus, ngram_range=(1,1)):
    vectorizer = CountVectorizer(min_df=1, ngram_range=ngram_range) # bow: bag of words詞袋模型; 宣告空的詞頻計算函數
    features = vectorizer.fit_transform(corpus) # 依corpus中的文件進行詞頻計算函數配適(fit)，並接著一一轉換(transform)corpus中的文件
# corpus
    return vectorizer, features #回傳配適好的詞頻計算函數，以及corpus中的文件詞項矩陣

# build bow vectorizer and get features
bow_vectorizer, bow_features = bow_extractor(CORPUS) # attention to bow_vectorized(傳回實的詞頻計算函數)

type(bow_features) # scipy.sparse.csr.csr_matrix

features = bow_features.todense() # Document-Term Matrix
print (features)

type(bow_vectorizer)

type(bow_features)

# extract features from new document using built vectorizer
new_doc_features = bow_vectorizer.transform(new_doc)
new_doc_features = new_doc_features.todense()
print (new_doc_features)

# print the feature names
feature_names = bow_vectorizer.get_feature_names()
print (feature_names)

import pandas as pd
def display_features(features, feature_names):
    df = pd.DataFrame(data=features,
                      columns=feature_names)
print (df)

display_features(features, feature_names)

display_features(new_doc_features, feature_names)

### 詞頻-逆文件頻率模型 TF-IDF Model
from sklearn.feature_extraction.text import TfidfTransformer
def tfidf_transformer(bow_matrix):
    transformer = TfidfTransformer(norm='l2',
                                   smooth_idf=True,
                                   use_idf=True)
    tfidf_matrix = transformer.fit_transform(bow_matrix)
    return transformer, tfidf_matrix

import numpy as np
import os
os.chdir('/Users/Vince/Google Drive/TextMining/III_scripts')
from feature_extractors import tfidf_transformer
feature_names = bow_vectorizer.get_feature_names()

# build tfidf transformer and show train corpus tfidf features
tfidf_trans, tdidf_features = tfidf_transformer(bow_features)
features = np.round(tdidf_features.todense(), 2)
display_features(features, feature_names)

# show tfidf features for new_doc using built tfidf transformer
nd_tfidf = tfidf_trans.transform(new_doc_features)
nd_features = np.round(nd_tfidf.todense(), 2)
display_features(nd_features, feature_names)

### B. 方法２使用sklearn.feature_extraction.text.TfidfTransformer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import numpy as np

corpus = ['this is the first document',
          'this document is the second document',
          'and this is the third one',
          'is this the first document']

vocabulary = ['this', 'document', 'first', 'is', 'second', 'the',
              'and', 'one']

pipe = Pipeline([('count', CountVectorizer(vocabulary=vocabulary)),
                 ('tfid', TfidfTransformer())]).fit(corpus)

pipe['count'].transform(corpus).toarray()

pipe['tfid'].idf_

pipe.transform(corpus).shape

### C. 文本數據正規化
# Use sklearn.preprocessing.Normalizer class to normalize data.
# https://scikit-learn.org/stable/modules/preprocessing.html
from __future__ import print_function
import numpy as np
from sklearn.preprocessing import Normalizer 

x = np.array([1, 2, 3, 4], dtype='float32').reshape(1,-1)

print("Before normalization: ", x)

options = ['l1', 'l2', 'max']

for opt in options:
    norm_x = Normalizer(norm=opt).fit_transform(x)
    print("After %s normalization: " % opt.capitalize(), norm_x)
    


### 文字資料相似性計算


### 文字資料集群分析案例
import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
import mpld3

import numpy as np
# load movie data
movie_data = pd.read_csv('movie_data.csv')
# view movie data
print (movie_data.head())

titles = movie_data['Title']
synopses = movie_data['Synopsis']
print (titles[:10]) #first 10 titles
print (synopses[0][:200]) #first 200 characters in first synopses (for'The Godfather')

### 停用詞、詞幹化與符號化
# load nltk's English stopwords as variable called 'stopwords'
stopwords = nltk.corpus.stopwords.words('english')

print (stopwords[:10])

# load nltk's SnowballStemmer as variabled 'stemmer'
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

# here I define a tokenizer and stemmer which returns the set of stems in the text that it is passed

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token 
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
           filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
# first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

#not super pythonic, no, not at all.
#use extend so it's a big flat list of vocab
totalvocab_stemmed = []
totalvocab_tokenized = []
for i in synopses:
    allwords_stemmed = tokenize_and_stem(i) #for each item in 'synopses', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list

allwords_tokenized = tokenize_only(i) # 324
totalvocab_tokenized.extend(allwords_tokenized) # 154529

vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
print ('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')

print (vocab_frame.head(25))

### 詞頻-逆文件頻率與文件相似性
from sklearn.feature_extraction.text import TfidfVectorizer

#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                   min_df=0.2, stop_words='english',
                                   use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

%time tfidf_matrix = tfidf_vectorizer.fit_transform(synopses) #fit the vectorizer to synopses
print(tfidf_matrix.shape)

terms = tfidf_vectorizer.get_feature_names()

from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)
print(dist)

### K-平均數集群
from sklearn.cluster import KMeans

num_clusters = 5

km = KMeans(n_clusters=num_clusters)

%time km.fit(tfidf_matrix)

clusters = km.labels_.tolist()

from sklearn.externals import joblib

#uncomment the below to save your model
#since I've already run my model I am loading from the pickle

#joblib.dump(km, 'doc_cluster.pkl')

km = joblib.load('doc_cluster.pkl')
clusters = km.labels_.tolist()
print(clusters)

films = { 'title': titles, 'synopsis': synopses, 'cluster': clusters}
frame = pd.DataFrame(films, index = [clusters] , columns = ['rank', 'title', 'cluster', 'genre'])

frame['cluster'].value_counts() #number of films per cluster (clusters from 0 to 4)

from __future__ import print_function
print("Top terms per cluster:")
print()
#sort cluster centers by proximity to centroid

order_centroids = km.cluster_centers_.argsort()[:, ::-1]
for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')

    for ind in order_centroids[i, :6]: #replace 6 with n words per cluster
        print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
        print() #add whitespace
        print() #add whitespace

        print("Cluster %d titles:" % i, end='')
        for title in frame.ix[i]['title'].values.tolist():
            print(' %s,' % title, end='')
        print() #add whitespace
        print() #add whitespace

print()
print()

### 多維尺度法
import os # for os.path.basename

import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.manifold import MDS

MDS()

# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

pos = mds.fit_transform(dist) # shape (n_components, n_samples)

xs, ys = pos[:, 0], pos[:, 1]

### 視覺化文件集群
#set up colors per clusters using a dict
cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3:'#e7298a', 4: '#66a61e'}

#set up cluster names using a dict
cluster_names = {0: 'Family, home, war',
                 1: 'Police, killed, murders',
                 2: 'Father, New York, brothers',
                 3: 'Dance, singing, love',
                 4: 'Killed, soldiers, captain'}

#some ipython magic to show the matplotlib plots inline
%matplotlib inline

#create data frame that has the result of the MDS plus the cluster numbe
rs and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles))

#group by cluster
groups = df.groupby('label')

# set up plot
fig, ax = plt.subplots(figsize=(17, 9)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
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

ax.legend(numpoints=1) #show legend with only 1 point

#add label in x,y position with the label as the film title
for i in range(len(df)):
    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)


plt.show() #show the plot


#uncomment the below to save the plot if need be
#plt.savefig('clusters_small_noaxes.png', dpi=200)

plt.close()












