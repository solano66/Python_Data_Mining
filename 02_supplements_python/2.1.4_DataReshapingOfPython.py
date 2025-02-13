'''
Collated by Ching-Shih Tsou 鄒慶士 博士 (Ph.D.) Distinguished Prof. at the Department of Mechanical Engineering/Director at the Center of Artificial Intelligence & Data Science (機械工程系特聘教授兼人工智慧暨資料科學研究中心主任), MCUT (明志科技大學); Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), NTUB (國立臺北商業大學)
Notes: This code is provided without warranty.
'''

### 2.1.4 Python語言資料變形
import pandas as pd
USArrests = pd.read_csv("./_data/USArrests.csv")
# 變數名稱調整
USArrests.columns = ['state', 'Murder', 'Assault',
'UrbanPop', 'Rape']

### Data reshaping by stack() and unstack()
# Wide to long
USArrests.set_index('state', inplace=True) # in_place=True 就地正法
USArrests_l = USArrests.stack()
USArrests_l

USArrests_l.unstack() # axis = 1, 50*4
USArrests_l.unstack(level=0) # transpose of above, 4*50
