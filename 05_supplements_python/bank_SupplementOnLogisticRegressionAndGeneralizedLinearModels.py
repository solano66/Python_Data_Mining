'''
Collated by Ching-Shih Tsou 鄒慶士 博士 (Ph.D.) Distinguished Prof. at the Department of Mechanical Engineering/Director at the Center of Artificial Intelligence & Data Science (機械工程系特聘教授兼人工智慧暨資料科學研究中心主任), MCUT (明志科技大學); Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), NTUB (國立臺北商業大學)
Notes: This code is provided without warranty.
'''

#### 5.1.5 羅吉斯迴歸分類與廣義線性模型補充範例
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

# credit = pd.read_csv('bank-additional-full_21.csv', header=0, sep=';')
credit = pd.read_csv('bank.csv', header=0)

credit.dtypes

print(credit.shape) # (41188, 21)
print(list(credit.columns)) # ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'emp_var_rate', 'cons_price_idx', 'cons_conf_idx', 'euribor3m', 'nr_employed', 'y']

# Show the counts of observations in each categorical bin using bars.
sns.countplot(x='y',data=credit, palette='hls') # try current_palette = sns.color_palette(); sns.palplot(current_palette); sns.palplot(sns.color_palette("hls", 8)); https://seaborn.pydata.org/tutorial/color_palettes.html
plt.show()

credit.isnull().sum()

# Here we only want to take following six predictors into account
# Customer job distribution
sns.countplot(y="job", data=credit)
plt.show()

# Customer marital status distribution
sns.countplot(x="marital", data=credit)
plt.show()

# Barplot for credit in default
sns.countplot(x="default", data=credit)
plt.show()

# Barplot for housing loan
sns.countplot(x="housing", data=credit)
plt.show()

# Barplot for personal loan
sns.countplot(x="loan", data=credit)
plt.show()

# Barplot for previous marketing campaign outcome
sns.countplot(x="poutcome", data=credit)
plt.show()

credit.drop(credit.columns[[0, 3, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19]], axis=1, inplace=True)

credit.dtypes

# Data Preprocessing (單熱編碼)
credit2 = pd.get_dummies(credit, columns =['job', 'marital', 'default', 'housing', 'loan', 'poutcome'])

credit2.dtypes

# Drop unknown class for all categorical variables
credit2.drop(credit2.columns[[12, 16, 18, 21, 24]], axis=1, inplace=True)
credit2.columns

# Check the independence between the independent variables
ax = sns.heatmap(credit2.corr())
#plt.setp(ax.get_xticklabels(which=('both'), rotation='vertical', fontsize=8))
plt.show()

# Split the data into training and test sets
X = credit2.iloc[:,1:]
y = credit2.iloc[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

X_train.shape

#### Logistic Regression Model
# Fit logistic regression to the training set
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

(3939+92)/np.sum(confusion_matrix)

print('Accuracy of logistic regression classifier on test set: {:.4f}'.format(classifier.score(X_test, y_test))) # 0.8980

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

#### Principal Component Analysis (to better vizualize the decision boundaries)
from sklearn.decomposition import PCA
X = credit2.iloc[:,1:]
y = credit2.iloc[:,0]
pca = PCA(n_components=2).fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(pca, y, random_state=0)

plt.figure(dpi=120)
plt.scatter(pca[y.values==0,0], pca[y.values==0,1], alpha=0.5, label='YES', s=2, color='navy')
plt.scatter(pca[y.values==1,0], pca[y.values==1,1], alpha=0.5, label='NO', s=2, color='darkorange')
plt.legend()
plt.title('Bank Marketing Data Set\nFirst Two Principal Components')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.gca().set_aspect('equal')
plt.show()

def plot_bank(X, y, fitted_model):
    plt.figure(figsize=(9.8,5), dpi=100)
    for i, plot_type in enumerate(['Decision Boundary', 'Decision Probabilities']):
        plt.subplot(1,2,i+1)
        mesh_step_size = 0.01  # step size in the mesh
        x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
        y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size), np.arange(y_min, y_max, mesh_step_size))
        if i == 0:
            Z = fitted_model.predict(np.c_[xx.ravel(), yy.ravel()])
        else:
            try:
                Z = fitted_model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1]
            except:
                plt.text(0.4, 0.5, 'Probabilities Unavailable', horizontalalignment='center',
                     verticalalignment='center', transform = plt.gca().transAxes, fontsize=12)
                plt.axis('off')
                break
        Z = Z.reshape(xx.shape)
        plt.scatter(X[y.values==0,0], X[y.values==0,1], alpha=0.8, label='YES', s=5, color='navy')
        plt.scatter(X[y.values==1,0], X[y.values==1,1], alpha=0.8, label='NO', s=5, color='darkorange')
        plt.imshow(Z, interpolation='nearest', cmap='RdYlBu_r', alpha=0.15, 
                   extent=(x_min, x_max, y_min, y_max), origin='lower')
        plt.title(plot_type + '\n' + 
                  str(fitted_model).split('(')[0]+ ' Test Accuracy: ' + str(np.round(fitted_model.score(X, y), 5)))
        plt.gca().set_aspect('equal');
    plt.tight_layout()
    plt.legend()
    plt.subplots_adjust(top=0.9, bottom=0.08, wspace=0.02)

model = LogisticRegression()
model.fit(X_train,y_train)
plot_bank(X_test, y_test, model)
plt.show()

#### Reference:
# Building A Logistic Regression in Python, Step by Step (https://datascienceplus.com/building-a-logistic-regression-in-python-step-by-step/)
# https://github.com/susanli2016/Machine-Learning-with-Python/find/master

