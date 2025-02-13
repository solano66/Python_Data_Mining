'''
Collated by Ching-Shih (Vince) Tsou 鄒慶士 博士 (Ph.D.) Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), NTUB (國立臺北商業大學); Founder of the Chinese Academy of R Software (CARS) (中華R軟體學會創會理事長); Founder of the Data Science and Business Applications Association of Taiwan (DSBA) (臺灣資料科學與商業應用協會創會理事長); the Chinese Association for Quality Assessment and Evaluation (CAQAE) (中華品質評鑑協會常務監事); the Chinese Society of Quality (CSQ) (中華民國品質學會大數據品質應用委員會主任委員
Notes: This code is provided without warranty.
'''

# We will predict the condition of a hydraulic rig, based on the sensor data provided by the following sensors:
# Pressure (5) 壓力柱
# Motor power (1) 馬達功率
# Volume flow (2) 體積流量
# Temperature (4) 溫度
# Vibration (1) 振動
# Cooling efficiency (virtual) (1) 冷卻效率
# Cooling power (virtual) (1) 冷卻功率
# Efficiency factor (1) 效率因子
# 
# Following is the information about the data set form the provider's readme file:
# 
# Condition monitoring of hydraulic systems
# =========================================
# 
# Abtract: The data set addresses the condition assessment of a hydraulic test rig 液壓鑽機 based on multi sensor data 多感測器資料. Four fault types are superimposed with several severity grades impeding selective quantification. 四種故障類型下各有幾個嚴重性等級
# 
# Source:
# Creator: ZeMA gGmbH, Eschberger Weg 46, 66121 Saarbrücken
# Contact: t.schneider@zema.de, s.klein@zema.de, m.bastuck@lmt.uni-saarland.de, info@lmt.uni-saarland.de
# 
# Data Type: Multivariate, Time-Series
# Task: Classification, Regression
# Attribute Type: Categorical, Real
# Area: CS/Engineering
# Format Type: Matrix
# Does your data set contain missing values? No
# 
# Number of Instances: 2205
# 
# Number of Attributes: 43680 (8x60 (1 Hz) + 2x600 (10 Hz) + 7x6000 (100 Hz))
# 
# Relevant Information:
# The data set was experimentally obtained with a hydraulic test rig. This test rig consists of a primary working and a secondary cooling-filtration circuit 
# which are connected via the oil tank [1], [2]. 
# The system cyclically repeats constant load cycles (duration 60 seconds) and measures process values such as pressures, volume flows and temperatures 
# while the condition of four hydraulic components (cooler, valve, pump and accumulator) is quantitatively varied. 
# 
# Attribute Information:
# The data set contains raw process sensor data (i.e. without feature extraction) which are structured as matrices (tab-delimited) 
# with the rows representing the cycles and the columns the data points within a cycle. The sensors involved are:
# Sensor		Physical quantity		Unit		Sampling rate
# PS1		Pressure			bar		100 Hz
# PS2		Pressure			bar		100 Hz
# PS3		Pressure			bar		100 Hz
# PS4		Pressure			bar		100 Hz
# PS5		Pressure			bar		100 Hz
# PS6		Pressure			bar		100 Hz
# EPS1		Motor power			W		100 Hz
# FS1		Volume flow			l/min		10 Hz
# FS2		Volume flow			l/min		10 Hz
# TS1		Temperature			°C		1 Hz
# TS2		Temperature			°C		1 Hz
# TS3		Temperature			°C		1 Hz
# TS4		Temperature			°C		1 Hz
# VS1		Vibration			mm/s		1 Hz
# CE		Cooling efficiency (virtual)	%		1 Hz
# CP		Cooling power (virtual)		kW		1 Hz
# SE		Efficiency factor		%		1 Hz
# 
# The target condition values are cycle-wise annotated in profile.txt (tab-delimited). As before, the row number represents the cycle number. The columns are
# 
# 1(0 in profile columns): Cooler condition / %:
# 	3: close to total failure
# 	20: reduced effifiency
# 	100: full efficiency
# 
# 2(1 in profile columns): Valve condition / %:
# 	100: optimal switching behavior
# 	90: small lag
# 	80: severe lag
# 	73: close to total failure
# 
# 3(2 in profile columns): Internal pump leakage:
# 	0: no leakage
# 	1: weak leakage
# 	2: severe leakage
# 
# 4(3 in profile columns): Hydraulic accumulator / bar:
# 	130: optimal pressure
# 	115: slightly reduced pressure
# 	100: severely reduced pressure
# 	90: close to total failure
# 
# 5(4 in profile columns): stable flag:
# 	0: conditions were stable
# 	1: static conditions might not have been reached yet
# 
# 

#### Import the libraries we will need 載入所需套件

# get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import xgboost as xgb # !conda install xgboost --y
import os


#### Import datafiles
dir_path = "../hydraulic_data/"


def get_files(dir_path, filename):
    return pd.read_csv(os.path.join(dir_path, filename), sep='\t', header=None)


#### Import all pressure sensors data 匯入所有壓力感測器資料
pressureFile1 = get_files(dir_path=dir_path, filename='PS1.txt') # (2205, 6000)
pressureFile2 = get_files(dir_path=dir_path, filename='PS2.txt')
pressureFile3 = get_files(dir_path=dir_path, filename='PS3.txt')
pressureFile4 = get_files(dir_path=dir_path, filename='PS4.txt')
pressureFile5 = get_files(dir_path=dir_path, filename='PS5.txt')
pressureFile6 = get_files(dir_path=dir_path, filename='PS6.txt')


#### Import volume flow files 匯入所有體積流量資料
volumeFlow1 = get_files(dir_path=dir_path, filename='FS1.txt') # (2205, 600)
volumeFlow2 = get_files(dir_path=dir_path, filename='FS2.txt')


#### Import temperature files 匯入溫度資料
temperature1 = get_files(dir_path=dir_path, filename='TS1.txt') # (2205, 60)
temperature2 = get_files(dir_path=dir_path, filename='TS2.txt')
temperature3 = get_files(dir_path=dir_path, filename='TS3.txt')
temperature4 = get_files(dir_path=dir_path, filename='TS4.txt')


#### import rest of the data files: pump efficiency, vibrations, cooling efficiency, coolin power, efficiency factor 匯入其餘資料：幫浦效率、振動、冷卻效率、冷卻功率、效率因子
pump1 = get_files(dir_path=dir_path, filename='EPS1.txt') # (2205, 6000)
vibration1 = get_files(dir_path=dir_path, filename='VS1.txt') # (2205, 60)
coolingE1 = get_files(dir_path=dir_path, filename='CE.txt') # (2205, 60)
coolingP1 = get_files(dir_path=dir_path, filename='CP.txt') # (2205, 60)
effFactor1 = get_files(dir_path=dir_path, filename='SE.txt') # (2205, 60)


#### Import Label data from profile file 從剖繪檔匯入標籤資料(y)
profile = get_files(dir_path=dir_path, filename='profile.txt') # (2205, 5)
profile.head()

#### Split the profile into relevent target labels 將剖繪資料分成對應的目標標籤(先處理y)
y_coolerCondition = pd.DataFrame(profile.iloc[:, 0]) # 冷卻器狀況 # (2205, 1)
y_valveCondition = pd.DataFrame(profile.iloc[:, 1]) # 閥門狀況
y_pumpLeak = pd.DataFrame(profile.iloc[:, 2]) # 幫浦(Wiki: 泵，或稱唧筒，又作幫浦，是一種用以增加流體的壓力，使加壓過的流體產生比平常狀況下更巨大的推進力量的裝置。泵運（Pumping）又稱泵送、抽運，是指泵的運作，可將液體或分子從一個位置移動到另一個位置。)洩漏
y_hydraulicAcc = pd.DataFrame(profile.iloc[:, 3]) # 液壓儲能器
y_stableFlag = pd.DataFrame(profile.iloc[:, 4]) # 穩定旗標


#### average the cycle data (提取各樣本的X均值特徵)
def mean_conversion(df):
    df1 = pd.DataFrame()
    df1 = df.mean(axis = 1)
    return df1


PS1 = pd.DataFrame(mean_conversion(pressureFile1))
PS1.columns = ['PS1']

PS2 = pd.DataFrame(mean_conversion(pressureFile2))
PS2.columns = ['PS2']

PS3 = pd.DataFrame(mean_conversion(pressureFile3))
PS3.columns = ['PS3']

PS4 = pd.DataFrame(mean_conversion(pressureFile4))
PS4.columns = ['PS4']

PS5 = pd.DataFrame(mean_conversion(pressureFile5))
PS5.columns = ['PS5']

PS6 = pd.DataFrame(mean_conversion(pressureFile6))
PS6.columns = ['PS6']

FS1 = pd.DataFrame(mean_conversion(volumeFlow1))
FS1.columns = ['FS1']

FS2 = pd.DataFrame(mean_conversion(volumeFlow2))
FS2.columns = ['FS2']

TS1 = pd.DataFrame(mean_conversion(temperature1))
TS1.columns = ['TS1']

TS2 = pd.DataFrame(mean_conversion(temperature2))
TS2.columns = ['TS2']

TS3 = pd.DataFrame(mean_conversion(temperature3))
TS3.columns = ['TS3']

TS4 = pd.DataFrame(mean_conversion(temperature4))
TS4.columns = ['TS4']

P1 = pd.DataFrame(mean_conversion(pump1))
P1.columns = ['P1']

VS1 = pd.DataFrame(mean_conversion(vibration1))
VS1.columns = ['VS1']

CE1 = pd.DataFrame(mean_conversion(coolingE1))
CE1.columns = ['CE1']

CP1 = pd.DataFrame(mean_conversion(coolingP1))
CP1.columns = ['CP1']

SE1 = pd.DataFrame(mean_conversion(effFactor1))
SE1.columns = ['SE1']


#### combine all dataframes
X = pd.concat([PS1, PS2, PS3, PS4, PS5, PS6, FS1, FS2, TS1, TS2, TS3, TS4, P1, VS1, CE1, CP1, SE1], axis=1) # (2205, 17)


#### Lets draw histogram of each value (pandas DataFrame簡便繪圖)
X.hist(bins=50, figsize=(20, 15))


#### let's get a correlation matix between various sensor parameters
corr_matrix = X.corr()

# plot heat map for correlation matrix (from https://seaborn.pydata.org/examples/many_pairwise_correlations.html)

# Generate a mask for the upper triangle
mask = np.zeros_like(corr_matrix, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=.3, center=0, annot=True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


#### Some of the features seems to be highly correlated. Let's normalize these parameters and do PCA 特徵高相關、進行主成份分析


# normalize the X values 主成份分析前須將Ｘ標準化
X_normalize = StandardScaler().fit_transform(X)

#apply PCA to visulaize the data in cluster
from sklearn.decomposition import PCA
#find explained variance and n_components

pca = PCA()
pca.fit(X_normalize)
dir(pca)
# Scree plot 陡坡圖決定主成份
plt.figure(figsize=(15, 10))
#plt.plot(pca.explained_variance_, linewidth=2)
plt.plot(pca.explained_variance_ratio_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_')
plt.ylim(-0.5, 2)
plt.title('PCA explained variance vs. n componets')


#### observe distribution when n_componets = 2 雙主成份下視覺化(樣本似乎分成4 ~ 5群)
pca_2 = PCA(n_components=2)
projected = pca_2.fit_transform(X_normalize)
print(X.shape) # (2205, 17)
print(projected.shape) # (2205, 2)
plt.figure(figsize=(15, 10))
plt.scatter(projected[:, 0], projected[:,1])
plt.xlabel('PCA (Dim=1)')
plt.ylabel('PCA (Dim=2)')
plt.title('Clustering of data when PCA dimension =2')
