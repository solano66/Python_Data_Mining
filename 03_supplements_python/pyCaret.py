### Collated by Ching-Shih Tsou 鄒慶士 博士 (Ph.D.) Distinguished Prof. at the Department of Mechanical Engineering/Director at the Center of Artificial Intelligence & Data Science (機械工程系特聘教授兼人工智慧暨資料科學研究中心主任), MCUT (明志科技大學); Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), NTUB (國立臺北商業大學)
### Notes: This code is provided without warranty.
### Datasets: sonar.csv, sonar.names

# compare machine learning algorithms on the sonar classification dataset
from pandas import read_csv
from pycaret.classification import setup # conda install -c conda-forge pycaret --y
from pycaret.classification import compare_models
# load the dataset
df = read_csv('sonar.csv', header=None)
# The task is to train a network to discriminate between sonar signals bounced off a metal cylinder and those bounced off a roughly cylindrical rock. 金屬圓筒 versus 圓柱形岩石

# set column names as the column number
n_cols = df.shape[1]
# df.columns[60] = ['Class'] # TypeError: Index does not support mutable operations
df.columns = ['attribute_'+str(i) for i in range(n_cols-1)] + ['Class']
# summarize the first few rows of data
print(df.head())

#### PyCaret for Comparing Machine Learning Models
# setup the dataset
grid = setup(data=df, target=df.columns[-1], html=False, silent=True, verbose=False)
# evaluate models and compare models
best = compare_models()

# Call the compare_models() function will also report a table of results summarizing all of the models that were evaluated and their performance.

# Matthews correlation coefficient (MCC). As an alternative measure unaffected by the unbalanced datasets issue, the Matthews correlation coefficient is a contingency matrix method of calculating the Pearson product-moment correlation coefficient between actual and predicted values. In terms of the entries of M, MCC reads as follows:

# MCC=TP⋅TN−FP⋅FN/sqrt((TP+FP)⋅(TP+FN)⋅(TN+FP)⋅(TN+FN))
# (worst value: –1; best value: +1)

# MCC is the only binary classification rate that generates a high score only if the binary predictor was able to correctly predict the majority of positive data instances and the majority of negative data instances.

# report the best model
print(best)

# We can then see the configuration of the model that was used, which looks like it used default hyperparameter values.

##### Tuning Machine Learning Models

# tune model hyperparameters on the sonar classification dataset
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier # This class implements a meta estimator that fits a number of randomized decision trees (a.k.a. extra-trees) on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
from pycaret.classification import setup
from pycaret.classification import tune_model
# load the dataset
df = read_csv('sonar.csv', header=None)
# set column names as the column number
n_cols = df.shape[1]
df.columns = [str(i) for i in range(n_cols)]
# setup the dataset
grid = setup(data=df, target=df.columns[-1], html=False, silent=True, verbose=False)
# tune model hyperparameters
best = tune_model(ExtraTreesClassifier(), n_iter=200, choose_better=True)

# We can tune model hyperparameters using the tune_model() function in the PyCaret library.

# The function takes an instance of the model to tune as input and knows what hyperparameters to tune automatically. A random search of model hyperparameters is performed and the total number of evaluations can be controlled via the “n_iter” argument.

# By default, the function will optimize the ‘Accuracy‘ and will evaluate the performance of each configuration using 10-fold cross-validation, although this sensible default configuration can be changed.

# A grid search is then performed reporting the performance of the best-performing configuration across the 10 folds of cross-validation and the mean accuracy.

# report the best model
print(best) # This processw is wiered !

#### Summary
# In this tutorial, you discovered the PyCaret Python open source library for machine learning.

# Specifically, you learned:

# PyCaret is a Python version of the popular and widely used caret machine learning package in R.
# How to use PyCaret to easily evaluate and compare standard machine learning models on a dataset.
# How to use PyCaret to easily tune the hyperparameters of a well-performing machine learning model.

#### References:
# https://datahub.io/machine-learning/sonar
# A Gentle Introduction to PyCaret for Machine Learning (https://machinelearningmastery.com/pycaret-for-machine-learning/)
# The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation (https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-019-6413-7#:~:text=MCC%20is%20the%20only%20binary,instances%20%5B80%2C%2097%5D.)
# Welcome to PyCaret (https://pycaret.gitbook.io/docs/)
# PyCaret for Classification: An Honest Review (https://towardsdatascience.com/pycaret-review-65cbe2f663bb)
