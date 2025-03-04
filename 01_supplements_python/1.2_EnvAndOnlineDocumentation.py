'''
Collated by Ching-Shih Tsou 鄒慶士 博士 (Ph.D.) Distinguished Prof. at the Department of Mechanical Engineering/Director at the Center of Artificial Intelligence & Data Science (機械工程系特聘教授兼人工智慧暨資料科學研究中心主任), MCUT (明志科技大學); Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), NTUB (國立臺北商業大學)
Notes: This code is provided without warranty.
'''

### 1.2 環境與輔助說明

## 環境與輔助說明{#sec1.4:bg}

# 在當前的環境中將符號"x"與物件168關聯起來
x = 168
# 同一環境中將符號"y"與物件2關聯起來
y = 2
# 符號"x"與"y"再組成"z"
z = x*y

# remove all globals() object
for name in dir():
    if not name.startswith('_'):
        del globals()[name]

# 查詢Python全域名稱空間中的物件(報表過長，請讀者自行執行)
globals()
# {'__name__': '__main__',
#  '__doc__': 'Automatically created module for IPython interactive environment',
#  '__package__': None,
#  '__loader__': None,
#  '__spec__': None,
#  '__builtin__': <module 'builtins' (built-in)>,
#  '__builtins__': <module 'builtins' (built-in)>,
#  '_ih': ['', 'globals()'],
#  '_oh': {},
#  '_dh': ['/Volumes/KINGSTON/GoogleDrive_HP/PragmaticBigDataAnalytics'],
#  'In': ['', 'globals()'],
#  'Out': {},
#  'get_ipython': <bound method InteractiveShell.get_ipython of <ipykernel.zmqshell.ZMQInteractiveShell object at 0x11a7ddb50>>,
#  'exit': <IPython.core.autocall.ZMQExitAutocall at 0x11a81ac40>,
#  'quit': <IPython.core.autocall.ZMQExitAutocall at 0x11a81ac40>,
#  '_': '',
#  '__': '',
#  '___': '',
#  'pd': <module 'pandas' from '/opt/anaconda3/lib/python3.8/site-packages/pandas/__init__.py'>,
#  'numpy': <module 'numpy' from '/opt/anaconda3/lib/python3.8/site-packages/numpy/__init__.py'>,
#  '_i': '',
#  '_ii': '',
#  '_iii': '',
#  '_i1': 'globals()'}


# 查詢Python局域名稱空間中的物件
locals()
# {'__name__': '__main__',
 # '__doc__': 'Automatically created module for IPython interactive environment',
 # '__package__': None,
 # '__loader__': None,
 # '__spec__': None,
 # '__builtin__': <module 'builtins' (built-in)>,
 # '__builtins__': <module 'builtins' (built-in)>,
 # '_ih': ['', 'globals()', 'locals()'],
 # '_oh': {1: {...}},
 # '_dh': ['/Volumes/KINGSTON/GoogleDrive_HP/PragmaticBigDataAnalytics'],
 # 'In': ['', 'globals()', 'locals()'],
 # 'Out': {1: {...}},
 # 'get_ipython': <bound method InteractiveShell.get_ipython of <ipykernel.zmqshell.ZMQInteractiveShell object at 0x11a7ddb50>>,
 # 'exit': <IPython.core.autocall.ZMQExitAutocall at 0x11a81ac40>,
 # 'quit': <IPython.core.autocall.ZMQExitAutocall at 0x11a81ac40>,
 # '_': {...},
 # '__': '',
 # '___': '',
 # 'pd': <module 'pandas' from '/opt/anaconda3/lib/python3.8/site-packages/pandas/__init__.py'>,
 # 'numpy': <module 'numpy' from '/opt/anaconda3/lib/python3.8/site-packages/numpy/__init__.py'>,
 # '_i': 'globals()',
 # '_ii': '',
 # '_iii': '',
 # '_i1': 'globals()',
 # '_1': {...},
 # '_i2': 'locals()'}

#  查詢Python全域與局域名稱空間中的變數(https://thepythonguru.com/python-builtin-functions/globals/)
# globals() # The globals() function returns a dictionary containing the variables defined in the global namespace. When globals() is called from a function or method, it returns the dictionary representing the global namespace of the module where the function or method is defined, not from where it is called.
# locals() # To access the local namespace use the locals() function.

# 顯示記憶體中現有的套件與資料集
dir()[:3]
# 讀入1947到1962年7個經濟變數資料集longley
import pandas as pd
# 指定環境為pandas套件，傳回其下的屬性與方法
dir(pd)[66:70]
fname1 = '~/cstsouMac/RBookWriting/bookdown-chinese-master/'
fname2 = './_data/longley.csv'
longley = pd.read_csv(fname1+fname2)
# 有看到資料集longley(報表過長，請讀者自行執行)
dir()

# 以drop()刪除名為"Unnamed: 0"列
# 刪除"列"時axis設定為1
longley = longley.drop(["Unnamed: 0"], axis = 1)
# 查看longley資料集中"GNP"資料
longley['GNP']

# 刪除在環境中的資料集或變數
del longley
# 卸載後沒見到資料集longley(報表過長，請讀者自行執行)
dir() 

# 以os.getcwd()查詢當前工作路徑，並儲存為iPAS字串物件
# 也可以用pwd指令查詢當前工作路徑
import os
iPAS = os.getcwd()

print(iPAS)

# 設定工作路徑為MAC OS下的家目錄
os.chdir('/Users/Vince')
# 取得當前的工作路徑
os.getcwd()
# 還原iPAS路徑
os.chdir(iPAS)

# 確定工作路徑已變更
os.getcwd()

# Python 線上輔助說明(報表過長，請讀者自行執行)
help('sys.warnoptions')
