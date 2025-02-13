'''
Collated by Ching-Shih Tsou 鄒慶士 博士 (Ph.D.) Distinguished Prof. at the Department of Mechanical Engineering/Director at the Center of Artificial Intelligence & Data Science (機械工程系特聘教授兼人工智慧暨資料科學研究中心主任), MCUT (明志科技大學); Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), NTUB (國立臺北商業大學)
Notes: This code is provided without warranty.
'''

from sklearn.datasets import fetch_20newsgroups
# import dataset
data = fetch_20newsgroups()
data.target_names

# Selected categories
categories = ['talk.politics.misc', 'talk.religion.misc', 'sci.med', 'sci.space', 'rec.autos']
# Create train and test dataset 
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)

print(train.data[5])

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB # Naive Bayes
from sklearn.pipeline import make_pipeline

# Create a pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB(alpha=1))
# alpha in multinomialNB represents the additive smoothing parameter. If it is 0, then there will be no smoothing.

# Why smoothing? Smoothing solves the zero probability problem in Naive Bayes algorithm, which is the problem of assigning probability equal to zero for every new data point in the test set.

# Fit the model with training set
model.fit(train.data, train.target)
#Predict labels for the test set
labels = model.predict(test.data)

from sklearn.metrics import confusion_matrix
import seaborn as sns # for plotting
import matplotlib.pyplot as plt # for plotting
# Create the confusion matrix
conf_mat = confusion_matrix(test.target, labels, normalize="true")

# Plot the confusion matrix
sns.heatmap(conf_mat.T, annot=True, fmt=".0%", cmap="cividis", xticklabels=train.target_names, yticklabels=train.target_names)
plt.xlabel("True label")
plt.ylabel("Predicted label")

#### Conclusion
# Naive Bayes classifiers are generally good for the initial baseline classification. They are fast, easy to implement and interpret. Having said so, they do not perform well for sophisticated models such as not having well-separated categories.

# Even though Naive Bayes is a simple classifier, it is used in many text-data based applications such as text classification (as we did in the example), sentiment analysis (i.e. understanding if the person thinks positively or negatively), and recommender systems (i.e. in collaborative filtering whether the customer is interested in a particular product group or not).

# To sum up we can list several advantages and disadvantages of Naive Bayes.

# ✅ Fast for training and prediction

# ✅ Easy to implement having only a few tunable parameters such as alpha

# ✅ Easy to interpret since they provide a probabilistic prediction

# ❌ Naive assumption of all predictors are independent do not hold in real life

# ❌ Zero probability problem can introduce wrong results when the smoothing technique wasn't used well.

# ❌ It can create highly biased estimations.

#### Reference:
# https://towardsdatascience.com/naive-bayes-algorithm-for-classification-bc5e98bff4d7