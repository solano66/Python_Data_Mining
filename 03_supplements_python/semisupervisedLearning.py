'''
Collated by Ching-Shih Tsou 鄒慶士 博士 (Ph.D.) Distinguished Prof. at the Department of Mechanical Engineering/Director at the Center of Artificial Intelligence & Data Science (機械工程系特聘教授兼人工智慧暨資料科學研究中心主任), MCUT (明志科技大學); Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), NTUB (國立臺北商業大學)
Notes: This code is provided without warranty.
'''

# prepare semi-supervised learning dataset
from sklearn.datasets import make_classification
# define dataset
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, random_state=1)

from sklearn.model_selection import train_test_split
# sklearn.model_selection.TimeSeriesSplit
# split into train (500) and test (500)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=1, stratify=y)

# split train into labeled (250 X_train_lab) and unlabeled (250 X_test_unlab)
X_train_lab, X_test_unlab, y_train_lab, y_test_unlab = train_test_split(X_train, y_train, test_size=0.50, random_state=1, stratify=y_train)

# summarize training set size, 250 labelled and 250 unlabeled
print('Labeled Train Set:', X_train_lab.shape, y_train_lab.shape)
print('Unlabeled Train Set:', X_test_unlab.shape, y_test_unlab.shape)
# summarize test set size
print('Test Set:', X_test.shape, y_test.shape)

# Next, we can establish a baseline in performance on the semi-supervised learning dataset using a supervised learning algorithm fit only on the labeled training data. 基線模型僅以有標籤的訓練樣本訓練模型
# This is important because we would expect a semi-supervised learning algorithm to outperform a supervised learning algorithm fit on the labeled data alone. If this is not the case, then the semi-supervised learning algorithm does not have skill. 期望半監督式學習的績效能超越基線模型

###  LogisticRegression 羅吉斯迴歸
# define model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
# fit model on labeled dataset
model.fit(X_train_lab, y_train_lab)

# make predictions on hold out test set
yhat1 = model.predict(X_test)
# calculate score for test set
from sklearn.metrics import accuracy_score
score1 = accuracy_score(y_test, yhat1)
# summarize score
print('Accuracy: %.3f' % (score1*100))

### Label Propagation for Semi-Supervised Learning半監督式學習之標籤傳遞法 
# The Label Propagation algorithm is available in the scikit-learn Python machine learning library via the LabelPropagation class.

from numpy import concatenate

# create the training dataset input 建立有標籤與無標籤混成的訓練資料集之輸入矩陣
X_train_mixed = concatenate((X_train_lab, X_test_unlab)) # (500, 2)
# create "no label" for unlabeled data
nolabel = [-1 for _ in range(len(y_test_unlab))] # 250
# recombine training dataset labels 建立有標籤與無標籤混成的訓練資料集之目標向量
y_train_mixed = concatenate((y_train_lab, nolabel)) # (500, ), 其中250個無標籤

# define model 建立半監督式學習模型
from sklearn.semi_supervised import LabelPropagation
model = LabelPropagation()
# fit model on training dataset
model.fit(X_train_mixed, y_train_mixed)
# make predictions on hold out test set
yhat2 = model.predict(X_test)
# calculate score for test set
score2 = accuracy_score(y_test, yhat2)
# summarize score
print('Accuracy: %.3f' % (score2*100))

# Another approach we can use with the semi-supervised model is to take the estimated labels for the training dataset and fit a supervised learning model. 另一種方法是將已估計標籤的訓練樣本，配適監督式學習模型
# Recall that we can retrieve the labels for the entire training dataset from the label propagation model as follows:

dir(model)
# get labels (含轉導標籤，前250個標籤與y_train_lab相同) for entire training dataset data
tran_labels = model.transduction_ # (500, ) 含轉導標籤
# define supervised learning model
model2 = LogisticRegression()
# fit supervised learning model on entire training dataset
model2.fit(X_train_mixed, tran_labels)
# make predictions on hold out test set
yhat3 = model2.predict(X_test)
# calculate score for test set
score3 = accuracy_score(y_test, yhat3)
# summarize score 表現最好的結果
print('Accuracy: %.3f' % (score3*100))

# In this case, we can see that this hierarchical approach of the semi-supervised model followed by supervised model achieves a classification accuracy of about 86.2 percent on the holdout dataset, even better than the semi-supervised learning used alone that achieved an accuracy of about 85.6 percent.

# Can you achieve better results by tuning the hyperparameters of the LabelPropagation model?

### Reference:
# Semi-Supervised Learning With Label Propagation (https://machinelearningmastery.com/semi-supervised-learning-with-label-propagation/)