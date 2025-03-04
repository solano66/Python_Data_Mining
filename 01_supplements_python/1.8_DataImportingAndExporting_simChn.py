'''
Collated by Prof. Ching-Shih Tsou 邹庆士 教授 (Ph.D.) at the IDS, CICD, and CISD of NTUB (台北商业大学资讯与决策科学研究所暨智能控制与决策研究室教授兼校务永续发展中心主任); at the ME Dept. and CAIDS of MCUT (2020年借调至明志科技大学机械工程系担任特聘教授兼人工智慧暨资料科学研究中心主任两年); the CSQ (2019年起任中华民国品质学会大数据品质应用委员会主任委员); the DSBA (2013年创立台湾资料科学与商业应用协会); and the CARS (2012年创立中华R软体学会)
Notes: This code is provided without warranty.
Dataset: C-MAPSS.zip
'''

#### 1.8.1 R 语言资料汇入及汇出案例
import os, fnmatch, glob
import pandas as pd
# 请先将C-MAPSS.zip解压缩为C-MAPSS文件夹(if necessary)
path = "./_data/C-MAPSS" # Please change directory to '_data' under codes_python

dir(os)

# 列出"./_data/C-MAPSS"下所有档名，观察命名规则
fnames = os.listdir(path)
fnames

# 抓取档名是train开头的档案，余请以次类推
pattern = "train*"
for entry in fnames:
    if fnmatch.fnmatch(entry, pattern): # Unix filename pattern matching, 用法fnmatch.fnmatch(filename, pattern)
        print(entry)
# 的确有四个train开头的档案

# 切换路径到"./_data/C-MAPSS"
os.chdir(path)

# 定义副档名为txt
ext = 'txt'

# 以for回圈抓取资料夹中train开头且副档名为txt的所有档名(格式化语法)，串列推导(list comprehension)语法是Python人群爱用的单行/列回圈
train_fnames = [i for i in glob.glob('train*.{}'.format(ext))] # Unix style pathname pattern expansion, glob.glob(pathname, *, recursive=False) Return a possibly-empty list of path names that match pathname

# glob.glob这个函式会回传所有符合'train*.txt'的档名清单。它只有一个参数"pathname"，定义了路径档名规则，可以是绝对路径，也可以是相对路径。

# 也可以简单地写成下面
# train_fnames = [i for i in glob.glob('train*')]

# 试着读第一个训练资料集看看 Get a good try here!
train0 = pd.read_csv(train_fnames[0], sep = ' ', header = None)
# 多抓了两个栏位 Some redundant columns read in.
train0

# 先定义栏位名称串列
colnames = ["unitNo", "cycles","opera1","opera2","opera3","T2","T24","T30","T50","P2","P15","P30","Nf","Nc","epr","Ps30","phi","NRf","NRc","BPR","farB","htBleed","Nf_dmd","PCNfr_dmd","W31","W32"]
# 限定栏位分隔字符、无标头、只读入26个栏位
train1 = pd.read_csv(train_fnames[1], sep = ' ', header = None, usecols = range(26), names = colnames)
# 正确无误读入资料档，且给予变数名称

# 试行成功后删除不必要的物件(Garabge collection)
del train0, train1

# 还是串列推导将所有训练集档案读入，并组织为原生资料结构串列(list)
trains = [pd.read_csv(f, sep = ' ', header = None, usecols = range(26), names = colnames) for f in train_fnames]

train0 = trains[0] 

train0.info()

# 将所有训练集档案合并(concatenate)成一个衍生资料结构资料框(DataFrame)
train_all = pd.concat(trains, axis=0) # encoding='ISO-8859-1'

train_all.info()

# 测试集与剩余寿命的读取请自行练习