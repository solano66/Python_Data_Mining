'''
Collated by Ching-Shih (Vince) Tsou 鄒慶士 博士 (Ph.D.) Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所暨智能控制與決策研究室教授), NTUB (國立臺北商業大學); Founder of the Chinese Academy of R Software (CARS) (中華R軟體學會創會理事長); Founder of the Data Science and Business Applications Association of Taiwan (DSBA) (臺灣資料科學與商業應用協會創會理事長); the Chinese Association for Quality Assessment and Evaluation (CAQAE) (中華品質評鑑協會常務監事); the Chinese Society of Quality (CSQ) (中華民國品質學會大數據品質應用委員會主任委員)
Notes: This code is provided without warranty.
'''

#### About Dataset
# 
#### Context
# 
# Access to safe drinking-water is essential to health, a basic human right and a component of effective policy for health protection. This is important as a health and development issue at a national, regional and local level. In some regions, it has been shown that investments in water supply and sanitation can yield a net economic benefit, since the reductions in adverse health effects and health care costs outweigh the costs of undertaking the interventions. 安全的飲用水對健康至為政要
# 
#### Content
# 
# The water_potability.csv file contains water quality metrics for 3276 different water bodies. 3276個不同水體
# 
#### 1. pH value: 酸鹼值
# PH is an important parameter in evaluating the acid–base balance of water. It is also the indicator of acidic or alkaline condition of water status. WHO has recommended maximum permissible limit of pH from 6.5 to 8.5. The current investigation ranges were 6.52–6.83 which are in the range of WHO standards.
# 
# 
#### 2. Hardness: 硬度
# Hardness is mainly caused by calcium and magnesium salts. These salts are dissolved from geologic deposits through which water travels. The length of time water is in contact with hardness producing material helps determine how much hardness there is in raw water. Hardness was originally defined as the capacity of water to precipitate soap caused by Calcium and Magnesium.
# 
# 
#### 3. Solids (Total dissolved solids - TDS): 總溶解固體
# Water has the ability to dissolve a wide range of inorganic and some organic minerals or salts such as potassium, calcium, sodium, bicarbonates, chlorides, magnesium, sulfates etc. These minerals produced un-wanted taste and diluted color in appearance of water. This is the important parameter for the use of water. The water with high TDS value indicates that water is highly mineralized. Desirable limit for TDS is 500 mg/l and maximum limit is 1000 mg/l which prescribed for drinking purpose.
# 
# 
#### 4. Chloramines: 氯胺
# Chlorine and chloramine are the major disinfectants used in public water systems. Chloramines are most commonly formed when ammonia is added to chlorine to treat drinking water. Chlorine levels up to 4 milligrams per liter (mg/L or 4 parts per million (ppm)) are considered safe in drinking water.
# 
# 
#### 5. Sulfate: 硫酸鹽
# Sulfates are naturally occurring substances that are found in minerals, soil, and rocks. They are present in ambient air, groundwater, plants, and food. The principal commercial use of sulfate is in the chemical industry. Sulfate concentration in seawater is about 2,700 milligrams per liter (mg/L). It ranges from 3 to 30 mg/L in most freshwater supplies, although much higher concentrations (1000 mg/L) are found in some geographic locations.
# 
# 
#### 6. Conductivity: 電導率
# Pure water is not a good conductor of electric current rather’s a good insulator. Increase in ions concentration enhances the electrical conductivity of water. Generally, the amount of dissolved solids in water determines the electrical conductivity. Electrical conductivity (EC) actually measures the ionic process of a solution that enables it to transmit current. According to WHO standards, EC value should not exceeded 400 μS/cm.
# 
# 
#### 7. Organic_carbon: 有機碳
# Total Organic Carbon (TOC) in source waters comes from decaying natural organic matter (NOM) as well as synthetic sources. TOC is a measure of the total amount of carbon in organic compounds in pure water. According to US EPA < 2 mg/L as TOC in treated / drinking water, and < 4 mg/Lit in source water which is use for treatment.
# 
# 
#### 8. Trihalomethanes: 三鹵甲晚
# THMs are chemicals which may be found in water treated with chlorine. The concentration of THMs in drinking water varies according to the level of organic material in the water, the amount of chlorine required to treat the water, and the temperature of the water that is being treated. THM levels up to 80 ppm is considered safe in drinking water.
# 
# 
#### 9. Turbidity: 濁度
# The turbidity of water depends on the quantity of solid matter present in the suspended state. It is a measure of light emitting properties of water and the test is used to indicate the quality of waste discharge with respect to colloidal matter. The mean turbidity value obtained for Wondo Genet Campus (0.98 NTU) is lower than the WHO recommended value of 5.00 NTU.
# 
# 
#### 10. Potability: 適飲性
# Indicates if water is safe for human consumption where 1 means Potable and 0 means Not potable.

# ------------------------------------------
#### import libraries

# ----------------


import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.offline as py
# py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout


# -------------------


data = pd.read_csv('./water_potability.csv') # (3276, 10)


# -----------------


data


# -----------------


data.isnull().sum()


# -----------------------------------------
#### EDA 探索式資料分析

# the correllation Before handling the null values

# -----------------


fig, ax = plt.subplots(figsize = (18,18))
sns.heatmap(data.corr(), ax = ax, annot = True)


# -------------------


fig,ax = plt.subplots(figsize=(8,8))
abs(data.corr().round(2)['Potability']).sort_values()[:-1].plot.barh(color='c')


# ------------------

(data['Potability']==0).sum() # 1998
data[data['Potability']==0][['ph','Sulfate','Trihalomethanes']].median()


# In[8]:

(data['Potability']==1).sum() # 1278
data[data['Potability']==1][['ph','Sulfate','Trihalomethanes']].median()


# In[9]:


data['ph'].fillna(value=data['ph'].median(),inplace=True)
data['Trihalomethanes'].fillna(value=data['Trihalomethanes'].median(),inplace=True)
data = data.dropna() # (2495, 10)


# In[10]:


data.isnull().sum()


# In[11]:


data.shape


# In[12]:


data.info()


# the correllation After handelling the null values

# In[13]:


fig,ax = plt.subplots(figsize=(8,8))
abs(data.corr().round(2)['Potability']).sort_values()[:-1].plot.barh(color='c')


# In[14]:


data.corr()['Potability'][:-1].sort_values().plot(kind='bar')


# In[15]:


trace = go.Pie(labels = ['Potable', 'Not Potable'], values = data['Potability'].value_counts(), 
               textfont=dict(size=15), opacity = 0.8,
               marker=dict(colors=['lightskyblue','gold'], 
                           line=dict(color='#000000', width=1.5)))


layout = dict(title =  'Distribution of Drinkable Water')
           
fig = dict(data = [trace], layout=layout)
py.iplot(fig)


# In[16]:


plt.figure(figsize = (15,10), tight_layout = True)

for i, feature in enumerate(data.columns):
    if feature != 'Potability':

        plt.subplot(3,3,i+1)
        sns.histplot(data = data, x =feature, palette = 'mako', hue = 'Potability',alpha = 0.5, element="step",hue_order=[1,0] )


# In[17]:


sns.pairplot(data = data,hue = 'Potability',palette='mako_r', corner=True)


# -----------------------------------------
# ## Data Splitting

# In[18]:


X = data.drop('Potability',axis=1).values
y = data['Potability'].values


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=40)


# -----------------------------
# ## Data Scalling

# In[20]:


scaler = MinMaxScaler()


# In[21]:


scaler.fit(X_train)


# In[22]:


X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[23]:


print('training shape : ',X_train.shape)
print('testing shape : ',X_test.shape)


# ------------------------------
# ## Modelling

# In[24]:


model = Sequential() # Initialising the ANN

model.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy')


# In[25]:


model.fit(x=X_train, 
          y=y_train, 
          epochs=300,
          validation_data=(X_test, y_test), verbose=1
          )


# In[26]:


model_loss = pd.DataFrame(model.history.history)


# In[27]:


model_loss.plot()


# In[28]:


y_pred = model.predict(X_test)
y_pred = [ 1 if y>=0.5 else 0 for y in y_pred ]


# In[29]:


from sklearn.metrics import classification_report,confusion_matrix


# In[30]:


print(classification_report(y_test,y_pred))


# In[31]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
print("Accuracy: "+ str(accuracy*100)+"%")


# In[32]:


model = Sequential()
model.add(Dense(units=16,activation='relu'))
model.add(Dense(units=8,activation='relu'))
model.add(Dense(units=4,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')


# In[33]:


model.fit(x=X_train, 
          y=y_train, 
          epochs=900,
          validation_data=(X_test, y_test), verbose=1
          )


# In[34]:


model_loss = pd.DataFrame(model.history.history)
model_loss.plot()


# In[35]:


y_pred = model.predict(X_test)
y_pred = [ 1 if y>=0.5 else 0 for y in y_pred ]


# In[36]:


print(classification_report(y_test,y_pred))


# In[37]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
print("Accuracy: "+ str(accuracy*100)+"%")


# In[38]:


model = Sequential()
model.add(Dense(units=10,activation='relu'))
model.add(Dense(units=8,activation='relu'))
model.add(Dense(units=8,activation='relu'))
model.add(Dense(units=6,activation='relu'))
model.add(Dense(units=6,activation='tanh'))
model.add(Dense(units=5,activation='relu'))
model.add(Dense(units=1,activation='tanh'))
model.compile(loss='binary_crossentropy', optimizer='adam')


# In[39]:


model.fit(x=X_train, 
          y=y_train, 
          epochs=500,
          validation_data=(X_test, y_test), verbose=1
          )


# In[40]:


model_loss = pd.DataFrame(model.history.history)
model_loss.plot()


# In[41]:


y_pred = model.predict(X_test)
y_pred = [ 1 if y>=0.5 else 0 for y in y_pred ]


# In[42]:


print(classification_report(y_test,y_pred))


# In[43]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
print("Accuracy: "+ str(accuracy*100)+"%")

