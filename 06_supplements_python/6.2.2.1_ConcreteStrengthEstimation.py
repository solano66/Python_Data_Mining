'''
Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the IDS and CICD (資訊與決策科學研究所暨智能控制與決策研究室), Director of the Center for Institutional and Sustainable Development (校務永續發展中心主任), NTUB (國立臺北商業大學)
Notes: This code is provided without warranty.
'''

# Create your first MLP in Keras
# from keras.models import Sequential
# from keras.layers import Dense
# import keras
# dir(keras)
# dir(keras.models)
# dir(keras.layers) # 'Cropping1D', 'Cropping2D', 'Cropping3D', 'CuDNNGRU', 'CuDNNLSTM'
#%%
import numpy as np
import pandas as pd
# fix random seed for reproducibility
np.random.seed(7)
# load pima indians dataset
concrete = pd.read_csv('./_data/concrete.csv', encoding='utf-8')
#%%
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() # 此處非標準化！而是[min=0, max=1]
scaler.fit(concrete)
concrete_norm = scaler.transform(concrete) # 真正做轉換
pd.DataFrame(concrete_norm).describe().T
#%%

X = concrete_norm[:,0:8]
y = concrete_norm[:,8]
#%%
#### 保留法資料切分
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=14)
#%%
#### 單層單一神經元
from sklearn.neural_network import MLPRegressor

model_mlp1 = MLPRegressor(random_state=0, activation='identity', hidden_layer_sizes=1) # one neuron in one layer
model_mlp1.fit(X_train, y_train)
mlp_score1=model_mlp1.score(X_train,y_train)

print('The coefficient of determination of the prediction (training set):', mlp_score1)
result1 = model_mlp1.predict(X_test)
#%%
import matplotlib.pyplot as plt
plt.scatter(y_test, result1)
plt.xlabel('Actual Strength')
plt.ylabel('Predicted Strength')
plt.plot([0,1],[0,1])
plt.title('Scoring on Test Data with One Neuron')

correlation1 = pd.DataFrame({'origin':y_test,'predict':result1})
correlation1.corr() # 0.250203
#%%
#### 單層五個神經元
model_mlp2 = MLPRegressor(random_state=0, activation='identity', hidden_layer_sizes=5) # five neurons in one layer
model_mlp2.fit(X_train, y_train)
mlp_score2=model_mlp2.score(X_train,y_train)

print('The coefficient of determination of the prediction (training set):',mlp_score2) # 0.38162487863469274 > 0.016718518158184814

result2 = model_mlp2.predict(X_test)

plt.scatter(y_test, result2)
plt.xlabel('Actual Strength')
plt.ylabel('Predicted Strength')
plt.plot([0,1],[0,1])
plt.title('Scoring on Test Data with Five Neurons')

correlation2 = pd.DataFrame({'origin':y_test,'predict':result2})
correlation2.corr() # 0.609792
#%%
#### 作業：把兩種結構MLP的權重係數及損失函數值抓出來
# 提示：dir(model_mlp1)

#### 兩層各五個神經元
model_mlp3 = MLPRegressor(random_state=0, activation='identity', hidden_layer_sizes=(5,5), max_iter=1000)
model_mlp3.fit(X_train, y_train)
mlp_score3 = model_mlp3.score(X_train,y_train)

print('The coefficient of determination of the prediction (training set):', mlp_score3) # 0.31284401538160866 < 0.38162487863469274

result3 = model_mlp3.predict(X_test)

plt.scatter(y_test, result3)
plt.xlabel('Actual Strength')
plt.ylabel('Predicted Strength')
plt.plot([0,1],[0,1])
plt.title('Scoring on Test Data with Five Neurons')

correlation3 = pd.DataFrame({'origin':y_test,'predict':result3})
correlation3.corr() # 0.595784

# %%
