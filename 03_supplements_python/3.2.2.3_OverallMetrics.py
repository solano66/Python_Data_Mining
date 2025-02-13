'''
Collated by Ching-Shih Tsou 鄒慶士 博士 (Ph.D.) Distinguished Prof. at the Department of Mechanical Engineering/Director at the Center of Artificial Intelligence & Data Science (機械工程系特聘教授兼人工智慧暨資料科學研究中心主任), MCUT (明志科技大學); Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), NTUB (國立臺北商業大學)
Notes: This code is provided without warranty.
'''

### 3.2.2.3_整體指標
import numpy as np

# 創建觀測與預測向量
observed = np.repeat(np.array(["No", "Yes"]), [21, 9], axis=0)
predicted = np.repeat(np.array(["No", "Yes"]), [22, 8], axis=0)

import random
observed = random.sample(observed.tolist(), len(observed))
predicted = random.sample(predicted.tolist(), len(predicted))

# 產生混淆矩陣(Confusion Matrix)
import pandas as pd
pd.crosstab(np.array(observed), np.array(predicted))

# 正確率
from sklearn.metrics import accuracy_score
accuracy_score(observed, predicted)

17/(17+6+7)

# Kappa係數/分數(Cohen提出)
from sklearn.metrics import cohen_kappa_score
print(cohen_kappa_score(observed, predicted))
