'''
Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the IDS, CICD, and CISD of NTUB (臺北商業大學資訊與決策科學研究所暨智能控制與決策研究室教授兼校務永續發展中心主任); at the ME Dept. and CAIDS of MCUT (2020年借調至明志科技大學機械工程系擔任特聘教授兼人工智慧暨資料科學研究中心主任兩年); the CSQ (2019年起任中華民國品質學會大數據品質應用委員會主任委員); the DSBA (2013年創立臺灣資料科學與商業應用協會); and the CARS (2012年創立中華R軟體學會) 
Notes: This code is provided without warranty.
'''

#### 3.2.2 分類模型績效指標
## ------------------------------------------------------------------------
import numpy as np
# 隨機產生觀測值向量
observed = np.random.choice(["No", "Yes"], 30, p=[2/3, 1/3])
# 觀測值向量一維次數分佈表
np.unique(observed, return_counts=True)
# 隨機產生預測值向量
predicted = np.random.choice(["No", "Yes"], 30, p=[2/3, 1/3])
# 預測值向量一維次數分佈表
np.unique(predicted, return_counts=True)
# 二維資料框
import pandas as pd
# 以原生字典物件創建兩欄pandas 資料框
res = pd.DataFrame({'observed': observed, 'predicted':
predicted})
print(res.head())

# pandas 套件建立混淆矩陣的兩種方式
# pandas 的crosstab() 交叉列表函數
print(pd.crosstab(res['observed'], res['predicted']))

# pandas 資料框的groupyby() 群組方法
print(res.groupby(['observed', 'predicted'])['observed'].
count())

# numpy 從觀測向量與預測向量計算正確率
print(np.mean(predicted == observed))

