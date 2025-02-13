'''
Collated by Ching-Shih Tsou 鄒慶士 博士 (Ph.D.) Distinguished Prof. at the Department of Mechanical Engineering/Director at the Center of Artificial Intelligence & Data Science (機械工程系特聘教授兼人工智慧暨資料科學研究中心主任), MCUT (明志科技大學); Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), NTUB (國立臺北商業大學); Founder of the Chinese Academy of R Software (CARS) (中華R軟體學會創會理事長) and the Data Science and Business Applications Association of Taiwan (DSBA) (臺灣資料科學與商業應用協會創會理事長); the Chinese Association for Quality Assessment and Evaluation (CAQAE) (中華品質評鑑協會常務監事); Director of the Committee on Big Data Quality Applications at the Chinese Society of Quality (CSQ) (中華民國品質學會大數據品質應用委員會主任委員)
Notes: This code is provided without warranty.
'''

#1) *Dummy Ecoding 虛擬編碼 & One Hot Encoding 單熱編碼
#2) *Label Encoding 標籤編碼
#3) Ordinal Encoding 順序編碼(視為順序尺度)
#4) *Helmert Encoding Helmert編碼(某水準下相依變量的均值，與所有其前水準下的均值相減)
#5) *Binary Encoding 二元編碼(用floor(log(base 2)n + 1)個特徵表達所有的組合情形)
#6) *Frequency Encoding 頻次編碼(用相對頻次當作編碼值)
#7) *Mean Encoding 均值編碼或目標編碼(各因子水準下目標變量的均值)
#8) *Weight of Evidence Encoding 證據權重編碼(二元目標變量勝率比對數值)
#9) Probability Ratio Encoding 機率比值編碼(就是證據權重編碼未取對數值)
#10) *Hashing Encoding 雜湊編碼(對應到有限個雜湊值，適合類別值非常多的時候)
#11) Backward Difference Encoding 後向差分編碼(Helmert encoding)
#12) Leave One Out Encoding 留一編碼(目標或均值編碼時，排除當前樣本的值以降低離群值的影響)
#13) James-Stein Encoding James-Stein編碼(特徵類別值下的目標變量均值與整體均值的加權平均)
#14) M-estimator Encoding M-估計子編碼(均值或目標編碼的正則化版本)
#15) Thermometer Encoder (http://www.bcp.psych.ualberta.ca/~mike/Pearl_Street/Dictionary/contents/T/thermcode.html#:~:text=Thermometer%20coding%20is%20usually%20used,would%20equal%20the%20encoded%20value.)
# Thermometer coding is one approach to representing information that is to be presented to an artificial neural network. Thermometer coding is usually used to represent a quantitative variable. Imagine some variable of this type that varies in value from 0 to 10. To thermometer code this variable, one would turn on a sequence of input units, where the length of the sequence would equal the encoded value. For instance, to represent a value of "2" the first two input units would be activated; to represent a value of "8", the first eight input units would be activated, and so on. It is called a thermometer code, because the input units resemble a thermometer placed on its side, with the "height" of the "mercury" in the thermometer representing the value of the input variable. Dawson and Zimmerman (2003) used a thermometer code to represent distances when a network was trained on the Piagetian balance scale task.

import pandas as pd
import numpy as np

data = {'Temperature' : ['Hot','Cold','Very Hot','Warm','Hot','Warm','Warm','Hot','Hot','Cold'],
        'Color' : ['Red', 'Yellow','Blue','Blue','Red','Yellow','Red','Yellow','Yellow','Yellow'],
        'Target' : [1,1,1,0,1,0,1,0,1,1]}

df = pd.DataFrame(data)

del data

df.head()

df.Temperature.value_counts(sort=False) # Four levels
df.Color.value_counts() # Three levels

#### Dummy Encoding 虛擬編碼 & One Hot Encoding 單熱編碼

df_dum = pd.get_dummies(df, prefix=['Temp'], columns=['Temperature'], drop_first=True) # prefix 'Temp' means Temp_Hot .....
df_dum

df_oh = pd.get_dummies(df, prefix=['Temp'], columns=['Temperature']) # Without the drop_first is the One Hot Encoding

from sklearn.preprocessing import OneHotEncoder

ohc = OneHotEncoder()
ohe = ohc.fit_transform(df.Temperature.values.reshape(-1,1)).toarray()

# Turn to DataFrame (columns: ['Temp_Cold', 'Temp_Hot', 'Temp_Very Hot', 'Temp_Warm'])
dfOneHot = pd.DataFrame(ohe, columns = ["Temp_"+str(ohc.categories_[0][i]) for i in range(len(ohc.categories_[0]))])

# Merge into DataFrame
dfh = pd.concat([df, dfOneHot], axis=1)

dfh

#### Label Encoding 標籤編碼從0開始

from sklearn.preprocessing import LabelEncoder # Step 1

# Cold: 0, Hot: 1, Very Hot: 2, Warm: 3
df['Temp_label_encoded'] = LabelEncoder().fit_transform(df.Temperature) # Steps 2&3&4

df

# Please notify the difference between numpy (lexicographical ordering) and pandas (appearance ordering) 字詞順序 vs. 外觀排序 (先出現先編)
df.loc[:, 'Temp_fatorize_encode'] = pd.factorize(df['Temperature'])[0].reshape(-1,1)

df

#### Ordinal Encoding 順序編碼從1開始(視為順序尺度)

# Ordinal encoding will assign values as ( Cold(1) <Warm(2)<Hot(3) <”Very Hot(4)). Usually, Ordinal Encoding is done starting from 1.

Temp_dict = {'Cold' : 1,
             'Warm' : 2,
             'Hot' : 3,
             'Very Hot' : 4} # getting hotter and hotter 越來越熱！

df['Temp_ordinal'] = df.Temperature.map(Temp_dict)
df

df.dtypes
# Temperature             object
# Color                   object
# Target                   int64
# Temp_label_encoded       int64
# Temp_fatorize_encode     int64
# Temp_ordinal             int64
# dtype: object

# drop some data for next experiment
df = df.drop(['Temp_label_encoded','Temp_fatorize_encode','Temp_ordinal'], axis=1)
df

#### Helmert Encoding (coding **unordered** factors) Helmert編碼(某水準下相依變量的均值，與所有其前水準下的均值相減)
# In this encoding, the mean of the dependent variable for a level is compared to the mean of the dependent variable over all previous levels. Hence, the name ‘reverse’ is used to differentiate from forward Helmert coding. 某水準下相依變量的均值，與所有其前水準下的均值相減

# Don't Forget to install the library conda install category_encoders
import category_encoders as ce # !conda install -c conda-forge category_encoders --y

encoder = ce.HelmertEncoder(cols=['Temperature'], drop_invariant=True) # drop_invariant: boolean for whether or not to drop columns with 0 variance.

dfh = encoder.fit_transform(df['Temperature'])

dir(encoder)
encoder.get_feature_names_out() # ['Temperature_0', 'Temperature_1', 'Temperature_2']

df = pd.concat([df, dfh], axis=1)
df # Hot, Cold, Very Hot, Warm編碼順序為依序觀測到的類別值

# drop some data for next experiment
df = df.drop(['Temperature_0','Temperature_1','Temperature_2'], axis=1)

#### Binary Encoding 二元編碼(用floor(log(base 2)n + 1)個特徵表達所有的組合情形)
# Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only floor(log(base 2)n + 1) features. In this example, we have three categories; thus, the binary encoded features will be three features. Compared to One Hot Encoding, this will require fewer feature columns (for 100 categories, One Hot Encoding will have 100 features, while for Binary encoding, we will need just seven features). 類別數越多效益越明顯，當有100個特徵時，100欄 vs. 7欄

import math
math.log2(3) + 1  # Two columns needed
np.floor(math.log2(100) + 1) # Only seven columns needed

# For Binary encoding, one has to follow the following steps:
# 1. The categories are first converted to numeric order starting from 1 (order is created as categories appear in a dataset and do not mean any ordinal nature)
# 2. Then those integers are converted into binary code, so for example, 3 becomes 011, 4 becomes 100
# 3. Then the digits of the binary number form separate columns.

encoder = ce.BinaryEncoder(cols=['Temperature'])
dfbin = encoder.fit_transform(df['Temperature'])
df = pd.concat([df, dfbin], axis=1)
df

# drop some data for next experiment
df = df.drop(['Temperature_0','Temperature_1','Temperature_2'], axis=1)

#### Frequency Encoding 頻次編碼(用相對頻次當作編碼值)
# Three-step for this :
# 1. Select a categorical variable you would like to transform
# 2. Group by the categorical variable and obtain counts of each category

fe = df.groupby('Temperature').size()/len(df)
fe # 2/10, 4/10, 1/10, 3/10

df.loc[:, 'Temp_freq_encode'] = df['Temperature'].map(fe)

df

# drop some data for next experiment
df = df.drop(['Temp_freq_encode'], axis=1)

#### Mean Encoding or Target Encoding 均值編碼或目標編碼(各因子水準下目標變量的均值)

# 類似標籤編碼，但是標籤值直接與目標變量相關，也就是說訓練樣本目標變數的均值決定各類別編碼值，此法不影響資料量，且有助於較快的訓練，但是在***過度配適方面惡名昭彰***，多數需要正則化與交叉驗證，有許多變形.
# Mean Encoding or Target Encoding is one viral encoding approach followed by Kagglers. There are many variations of this. Here I will cover the basic version and smoothing version. Mean encoding is similar to label encoding, except here labels are correlated directly with the target. For example, in mean target encoding for each category in the feature label is decided with the mean value of the target variable on training data. This encoding method brings out the relation between similar categories, but the connections are bounded within the categories and target itself. The advantages of the mean target encoding are that it does not affect the volume of the data and helps in faster learning. Usually, Mean encoding is notorious for over-fitting; thus, a regularization with cross-validation or some other approach is a must on most occasions.

# 1. Select a categorical variable you would like to transform.
# 2. Group by the categorical variable and obtain aggregated sum over the “Target” variable. 依類別變量群組後，各組加總目標變數值
# 3. Group by the categorical variable and obtain aggregated count over “Target” variable 依類別變量群組後，各組計算樣本數
# 4. Divide the step 2 / step 3 results and join it back with the train. 步驟2除以步驟3為各類別編碼值(如目標變數為二元，即為群組後計算均值)

mean_encode = df.groupby('Temperature')['Target'].mean()
print(mean_encode)
df.loc[:, 'Temperature_mean_enc'] = df['Temperature'].map(mean_encode)
df

# drop some data for next experiment
df = df.drop(['Temperature_mean_enc'], axis=1)

# another variation ways 另一種變形 - 平滑化

# Compute the global mean
mean = df['Target'].mean()

# Compute the number of values and the mean of each group
agg = df.groupby('Temperature')['Target'].agg(['count', 'mean']) # 第一種編碼數值

counts = agg['count']
means = agg['mean'] # 第一種編碼數值

weight = 100

# Compute the 'smoothed' mean
smooth = (counts * means + weight * mean) / (counts + weight)

# Replace each value by the according smoothed mean
print(smooth)
df.loc[:, 'Temperature_smean_enc'] = df['Temperature'].map(smooth) # 各類別編碼值較為接近(smoothing之意)
df

# drop some data for next experiment
df = df.drop(['Temperature_smean_enc'], axis=1)

#### Weight of Evidence Encoding 證據權重編碼(二元目標變量勝率比對數值)
# Weight of Evidence (WoE) measures the “strength” of a grouping technique to separate good and bad 區分好與壞的強度. This method was developed primarily to build a predictive model to evaluate the risk of loan default in the credit and financial industry. Weight of evidence (WOE) measures how much the evidence supports or undermines a hypothesis 衡量支持或反駁假設的強度.

# WoE will be 0 if the P(Goods) / P(Bads) = 1. That is, if the outcome is random for that group 好壞參半WoE為零. If P(Bads) > P(Goods) the odds ratio will be < 1 好低於壞WoE小於零 and the WoE will be < 0; if, on the other hand, P(Goods) > P(Bads) in a group, then WoE > 0 好高於壞WoE大於零.

# We calculate probability of target = 1 i.e. Good = 1 for each category (Class label {0, 1} to probability by averaging)
woe_df = df.groupby('Temperature')['Target'].mean() # (4,)
# For the method 'rename'
woe_df = pd.DataFrame(woe_df) # Series (4,) to DataFrame (4, 1)

# Rename the column name to 'Good' to keep it consistent with formula for easy understanding
woe_df = woe_df.rename(columns = {'Target' : 'Good'})

# Calculate Bad probability which is 1 - Good probability
woe_df['Bad'] = 1 - woe_df.Good
woe_df

# We need to add a small value to avoid divide by zero in denominator 避免除以零
woe_df['Bad'] = np.where(woe_df['Bad'] == 0, 0.000001, woe_df['Bad']) # Like the 'ifelse' in R

# https://numpy.org/doc/stable/reference/generated/numpy.where.html
a = np.arange(10)
np.where(a < 5, a, 10*a)
#########################

# compute the WoE 勝率比對數值
woe_df['WoE'] = np.log(woe_df.Good / woe_df.Bad)
woe_df

# Map the WoE values back to each row of data-frame
df.loc[:, 'WoE_Encode'] = df['Temperature'].map(woe_df['WoE'])
df

# drop some data for next experiment
df = df.drop(['WoE_Encode'], axis=1)

#### Probability Ratio Encoding 機率比值編碼(就是證據權重編碼未取對數值)
# Probability Ratio Encoding is similar to Weight Of Evidence(WoE) 類似證據編碼權重, with the only difference is the only ratio of good and bad probability is used. For each label, we calculate the mean of target=1, that is, the probability of being 1 ( P(1) ), and also the probability of the target=0 ( P(0) ). And then, we calculate the ratio P(1)/P(0) and replace the labels with that ratio. We need to add a minimal value with P(0) to avoid any divide by zero scenarios where for any particular category, there is no target=0. 各類P(1)/P(0)為編碼值，分母需加上一微小值

# We calculate probability of target = 1 i.e. Good = 1 for each category
pr_df = df.groupby('Temperature')['Target'].mean() # 就是計算各類Good的機率
pr_df = pd.DataFrame(pr_df)

# Rename the column name to 'Good' to keep it consistent with formula for easy understanding
pr_df = pr_df.rename(columns = {'Target' : 'Good'})

# Calculate Bad probability which is 1- Good probability
pr_df['Bad'] =  1 - pr_df.Good

# We neef to add a small value to avoid divide by zero in denominator
pr_df['Bad'] = np.where(pr_df['Bad'] == 0, 0.000001, pr_df['Bad'])

# compute the Probability Ratio
pr_df['PR'] = pr_df.Good / pr_df.Bad
pr_df

# MAp the Probability Ratio values back to each row of data-frame
df.loc[:, 'PR_Encode'] = df['Temperature'].map(pr_df['PR'])
df

# drop some data for next experiment
df = df.drop(['Color','PR_Encode'], axis=1)

#### Hashing 雜湊編碼(對應到有限個雜湊值，適合類別值非常多的時候)
# Hashing converts categorical variables to a higher dimensional space of integers, where
# the distance between two vectors of categorical variables is approximately maintained by the transformed numerical dimensional space. With Hashing, the number of dimensions will be far less than the number of dimensions with encoding like One Hot Encoding. 雜湊編碼後的維度遠小於單熱編碼 This method is advantageous when the cardinality of categorical is very high. 此法適合類別值非常多的時候

#### Backward Difference Encoding 後向差分編碼(Helmert encoding)
# In backward difference coding, the mean of the dependent variable for a level is
# compared with the mean of the dependent variable for the prior level. This type of coding may be useful for a nominal or an ordinal variable. 適合名目或順序尺度變量
# This technique falls under the contrast coding system for categorical features. 屬於對比編碼系統 A feature of K categories, or levels, usually enters a regression as a sequence of K-1 dummy variables. K類特徵通常有K-1個虛擬變量

#### Leave One Out Encoding 留一編碼(目標或均值編碼時，排除當前樣本的值以降低離群值的影響)
# This is very similar to target encoding but excludes the current row’s target when calculating the mean target for a level to reduce outliers. 目標或均值編碼時，排除當前樣本的值以降低離群值的影響

#### James-Stein Encoding James-Stein編碼(特徵類別值下的目標變量均值與整體均值的加權平均)
# For feature value, the James-Stein estimator returns a weighted average of: 下面兩者的加權平均值
# 1. The mean target value for the observed feature value. 特徵類別值下的目標變量均值 2. The mean target value (regardless of the feature value). 不分特徵類別值下的目標變量均值
# The James-Stein encoder shrinks the average toward the overall average. It is a target based encoder.目標變量為基礎的編碼器 James-Stein estimator has, however, one practical limitation — it was defined only for normal distributions. 基於常態分佈

#### M-estimator Encoding M-估計子編碼(均值或目標編碼的正則化版本)
# M-Estimate Encoder is a simplified version of Target Encoder. It has only one hyper-
# parameter — m, which represents the power of regularization. The higher the value of m results, into stronger the shrinking. Recommended values for m is in the range of 1 to 100. 目標編碼的正則化版本，透過超參數 m (介於1到100) 調節係數縮減程度

# FAQ:
# I received many queries related to using or how one can treat the test data when there is no target. I am adding a Faq section here, which I hope would assist.
# Faq 01: Which method should I use?
# Answer: There is no single method that works for every problem or dataset. You may have to try a few to see, which gives a better result. The general guideline is to refer to the cheat sheet shown at the end of the article.

# Answer: We need to use the mapping values created at the time of training. This process is the same as scaling or normalization, where we use the train data to scale or normalize the test data. Then map and use the same value in testing time pre- processing. We can even create a dictionary for each category and mapped value and then use the dictionary at testing time. Here I am using the mean encoding to explain this.

#### Training Time 訓練時間

mean_encode = df.groupby('Temperature')['Target'].mean()
print(mean_encode)
df.loc[:, 'Temperature_mean_enc'] = df['Temperature'].map(mean_encode)
df

#### Testing Time 測試時間

# Encoded values from mean encoding
print(mean_encode)

# test data without the target
test_data = {'Temperature' : ['Cold','Very Hot','Warm','Hot','Hot','Hot','Cold','Cold']}
dft = pd.DataFrame(test_data,columns = ['Temperature'])

# map temperature using map data for mean encoding created during training
dft['Temperature_mean_enc'] = dft['Temperature'].map(mean_encode)
dft

#### Conclusion 結論
# It is essential to understand that all these encodings do not work well in all situations or for every dataset for all machine learning models. Data Scientists still need to experiment and find out which works best for their specific case. 實驗、實驗、再實驗 If test data has different classes, some of these methods won’t work as features won’t be similar. There are few benchmark publications by research communities, but it’s not conclusive which works best. My recommendation will be to try each of these with the smaller datasets and then decide where to focus on tuning the encoding process. You can use the below cheat-sheet as a guiding tool.

#### for more detail go here:
# https://towardsdatascience.com/all-about-categorical-variable-encoding-305f3361fd02


