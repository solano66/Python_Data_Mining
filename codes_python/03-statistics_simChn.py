'''
Collated by Prof. Ching-Shih Tsou 邹庆士 教授 (Ph.D.) at the IDS, CICD, and CISD of NTUB (台北商业大学资讯与决策科学研究所暨智能控制与决策研究室教授兼校务永续发展中心主任); at the ME Dept. and CAIDS of MCUT (2020年借调至明志科技大学机械工程系担任特聘教授兼人工智慧暨资料科学研究中心主任两年); the CSQ (2019年起任品质学会大数据品质应用委员会主任委员); the DSBA (2013年创立台湾资料科学与商业应用协会); and the CARS (2012年创立中华R软体学会) 
Notes: This code is provided without warranty.
'''

#### 3.2.2 分类模型绩效指标
## ------------------------------------------------------------------------
import numpy as np
# 随机产生观测值向量
observed = np.random.choice(["No", "Yes"], 30, p=[2/3, 1/3])
# 观测值向量一维次数分布表
np.unique(observed, return_counts=True)
# 随机产生预测值向量
predicted = np.random.choice(["No", "Yes"], 30, p=[2/3, 1/3])
# 预测值向量一维次数分布表
np.unique(predicted, return_counts=True)
# 二维资料框
import pandas as pd
# 以原生字典物件创建两栏pandas 资料框
res = pd.DataFrame({'observed': observed, 'predicted':
predicted})
print(res.head())

# pandas 套件建立混淆矩阵的两种方式
# pandas 的crosstab() 交叉列表函数
print(pd.crosstab(res['observed'], res['predicted']))

# pandas 资料框的groupyby() 群组方法
print(res.groupby(['observed', 'predicted'])['observed'].
count())

# numpy 从观测向量与预测向量计算正确率
print(np.mean(predicted == observed))

