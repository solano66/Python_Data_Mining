'''
Collated by Ching-Shih Tsou 鄒慶士 博士 (Ph.D.) Distinguished Prof. at the Department of Mechanical Engineering/Director at the Center of Artificial Intelligence & Data Science (機械工程系特聘教授兼人工智慧暨資料科學研究中心主任), MCUT (明志科技大學); Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), NTUB (國立臺北商業大學)
Notes: This code is provided without warranty.
'''

### 3.2.3_模型績效視覺化
import numpy as np
from sklearn.metrics import roc_curve, auc # area under curve
actual = np.append(np.repeat(0,10), np.repeat(1,10))
predProb = np.array([0.9,0.8,0.6,0.55,0.54,0.51,0.4,0.38,0.34,0.3,0.7,0.53,0.52,0.505,0.39,0.37,0.36,0.35,0.33,0.1])

import pandas as pd
df = pd.DataFrame(data=np.concatenate((actual.reshape(-1,1), predProb.reshape(-1,1)), axis=1), columns=['actual','predProb'])
df.to_csv('roc_df.csv')

tpr, fpr, _ = roc_curve(actual, predProb) # 輸入真實標籤與陽例的預測機率值(好像反了！)
# fpr, tpr, thresholds = roc_curve(actual, predProb) # 傳回fpr, tpr與計算所需之陰陽切分門檻值
roc_auc = auc(fpr, tpr)

#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt

fig = plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # 45 degree straight line
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.legend(loc="lower right")
#fig.savefig('/tmp/roc.png')
plt.show()

#http://blog.changyy.org/2017/09/python-roc-receiver-operating.html