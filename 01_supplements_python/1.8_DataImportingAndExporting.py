'''
Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the IDS, CICD, and CISD of NTUB (臺北商業大學資訊與決策科學研究所暨智能控制與決策研究室教授兼校務永續發展中心主任); at the ME Dept. and CAIDS of MCUT (2020年借調至明志科技大學機械工程系擔任特聘教授兼人工智慧暨資料科學研究中心主任兩年); the CSQ (2019年起任中華民國品質學會大數據品質應用委員會主任委員); the DSBA (2013年創立臺灣資料科學與商業應用協會); and the CARS (2012年創立中華R軟體學會)
Notes: This code is provided without warranty.
Dataset: C-MAPSS.zip
'''

#### 1.8.1 R 語言資料匯入及匯出案例
import os, fnmatch, glob
import pandas as pd
# 請先將C-MAPSS.zip解壓縮為C-MAPSS文件夾(if necessary)
path = "./_data/C-MAPSS" # Please change directory to codes_python (where there is a folder '_data' under it)

dir(os)

# 列出"./_data/C-MAPSS"下所有檔名，觀察命名規則
fnames = os.listdir(path)
fnames

# 抓取檔名是train開頭的檔案，餘請以次類推
pattern = "train*"
for entry in fnames:
    if fnmatch.fnmatch(entry, pattern): # Unix filename pattern matching, 用法fnmatch.fnmatch(filename, pattern)
        print(entry)
# 的確有四個train開頭的檔案

# 切換路徑到"./_data/C-MAPSS"
os.chdir(path)

# 定義副檔名為txt
ext = 'txt'

# 以for迴圈抓取資料夾中train開頭且副檔名為txt的所有檔名(格式化語法)，串列推導(list comprehension)語法是Python人群愛用的單行/列迴圈
train_fnames = [i for i in glob.glob('train*.{}'.format(ext))] # Unix style pathname pattern expansion, glob.glob(pathname, *, recursive=False) Return a possibly-empty list of path names that match pathname

# glob.glob這個函式會回傳所有符合'train*.txt'的檔名清單。它只有一個參數"pathname"，定義了路徑檔名規則，可以是絕對路徑，也可以是相對路徑。

# 也可以簡單地寫成下面
# train_fnames = [i for i in glob.glob('train*')]

# 試著讀第一個訓練資料集看看 Get a good try here!
train0 = pd.read_csv(train_fnames[0], sep = ' ', header = None)
# 多抓了兩個欄位 Some redundant columns read in.
train0

# 先定義欄位名稱串列
colnames = ["unitNo", "cycles","opera1","opera2","opera3","T2","T24","T30","T50","P2","P15","P30","Nf","Nc","epr","Ps30","phi","NRf","NRc","BPR","farB","htBleed","Nf_dmd","PCNfr_dmd","W31","W32"]
# 限定欄位分隔字符、無標頭、只讀入26個欄位
train1 = pd.read_csv(train_fnames[1], sep = ' ', header = None, usecols = range(26), names = colnames)
# 正確無誤讀入資料檔，且給予變數名稱

# 試行成功後刪除不必要的物件(Garabge collection)
del train0, train1

# 還是串列推導(單列/行迴圈寫法)將所有訓練集檔案讀入，並組織為原生資料結構串列(list)
trains = [pd.read_csv(f, sep = ' ', header = None, usecols = range(26), names = colnames) for f in train_fnames] # train_fnames is an iterable (可迭代物件、集合容器)

train0 = trains[0] 

train0.info()

# (optional!)將所有訓練集檔案合併(concatenate)成一個衍生資料結構資料框(DataFrame)
train_all = pd.concat(trains, axis=0) # encoding='ISO-8859-1'
# cbind() or rbind in R

train_all.info()

# 測試集與剩餘壽命的讀取請自行練習