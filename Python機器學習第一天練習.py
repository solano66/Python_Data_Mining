
from sklearn import datasets
import numpy as np

#Loading and preprocessing the data
#Loading the Iris dataset from scikit-learn. Here, the third column represents the petal length, and the fourth column the petal width of the flower samples.
iris = datasets.load_iris()
iris.keys() # Bunch物件的鍵
iris.feature_names # 檢視屬性名稱
iris.target_names # 檢視目標屬性名稱
iris.data.shape


X = iris.data[:, :]
y = iris.target


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size = 0.25, random_state = 48)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train, y_train)
knn.get_params()

y_predicted = knn.predict(X_test)

y_test == y_predicted

np.mean(y_test == y_predicted)
acc = np.mean(y_test == y_predicted) * 100
print("正確率為{0:.1f}%".format(acc))

from sklearn.model_selection import cross_val_score
help(cross_val_score)
cross_val_score(knn, X_std, y, scoring = "accuracy")

# cv for different k
avg = []
det = []
params = list(range(1,26))
for k in params:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_std, y, scoring='accuracy')
    avg.append(np.mean(scores))
    det.append(scores)
