# --- SECTION 1 ---
# Libraries and data loading
import numpy as np
import pandas as pd

from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics

# import pyreadr # conda install -c conda-forge pyreadr==0.3.3 --y (Python < 3.9), 0.4.4 downgraded to v0.3.3
# data = pyreadr.read_r('./data/creditcard.Rdata')['creditcard']
# data = pd.read_csv('creditcard.csv', dtype={'Class':'category'})
data = pd.read_csv('./data/creditcard.csv')
data.drop(['Unnamed: 0'], axis=1, inplace=True)

data.dtypes # Data has already been desensitized.
# Time       float64
# V1         float64
# V2         float64
# V3         float64
# V4         float64
# V5         float64
# V6         float64
# V7         float64
# V8         float64
# V9         float64
# V10        float64
# V11        float64
# V12        float64
# V13        float64
# V14        float64
# V15        float64
# V16        float64
# V17        float64
# V18        float64
# V19        float64
# V20        float64
# V21        float64
# V22        float64
# V23        float64
# V24        float64
# V25        float64
# V26        float64
# V27        float64
# V28        float64
# Amount     float64
# Class     category -> int64 (okay for the folllowing execution)

tmp = data.head()
tmp1 = data.describe(include='all') # A class imbalance (heavily skewed/ill-distribued) binary classification problem 


np.random.seed(123456)

data.Time = (data.Time-data.Time.min())/data.Time.std() # Standardized by ourselves
data.Amount = (data.Amount-data.Amount.mean())/data.Amount.std()

# Train-Test slpit of 70%-30%
x_train, x_test, y_train, y_test = train_test_split(
        data.drop('Class', axis=1).values, data.Class.values, test_size=0.3)

# --- SECTION 2 ---
# Ensemble evaluation
base_classifiers = [('DT', DecisionTreeClassifier(max_depth=5)),
                ('NB', GaussianNB()),
                ('LogisReg', LogisticRegression()),
                ('DT2', DecisionTreeClassifier(max_depth=3)),
                ('DT3', DecisionTreeClassifier(max_depth=8))]

#### Non-generative ensemble
ensemble = VotingClassifier(base_classifiers)
ensemble.fit(x_train, y_train) # Can not be fitted under sklearn 0.21.x

print('Voting f1', metrics.f1_score(y_test, ensemble.predict(x_test))) # .to_numpy()
# ValueError: pos_label=1 is not a valid label: array(['0', '1'], dtype='<U1')
# Voting f1 0.8294573643410852
print('Voting recall', metrics.recall_score(y_test, ensemble.predict(x_test))) # Voting recall 0.7867647058823529



# --- SECTION 3 ---
# Filter features according to their correlation to the target (過濾法)
np.random.seed(123456)
threshold = 0.1

correlations = data.corr()['Class'].drop('Class') # Pick out the column 'Class' and remove the row 'Class'
fs = list(correlations[(abs(correlations)>threshold)].index.values)
fs.append('Class')
data = data[fs]

x_train, x_test, y_train, y_test = train_test_split(
        data.drop('Class', axis=1).values, data.Class.values, test_size=0.3)

ensemble = VotingClassifier(base_classifiers)
ensemble.fit(x_train, y_train)

print('Voting f1', metrics.f1_score(y_test, ensemble.predict(x_test))) # Voting f1 0.8275862068965517 (degrade a little bit!)
print('Voting recall', metrics.recall_score(y_test, ensemble.predict(x_test))) # Voting recall 0.7941176470588235 (improved!)
