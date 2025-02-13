'''
Collated by Ching-Shih (Vince) Tsou 鄒慶士 博士 (Ph.D.) Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所暨智能控制與決策研究室教授), NTUB (國立臺北商業大學); Founder of the Chinese Academy of R Software (CARS) (中華R軟體學會創會理事長); Founder of the Data Science and Business Applications Association of Taiwan (DSBA) (臺灣資料科學與商業應用協會創會理事長); the Chinese Association for Quality Assessment and Evaluation (CAQAE) (中華品質評鑑協會常務監事); the Chinese Society of Quality (CSQ) (中華民國品質學會大數據品質應用委員會主任委員)
Notes: This code is provided without warranty.
'''

# Partial Least Square (PLS) regression is one of the workhorses of chemometrics applied to spectroscopy. PLS can successfully deal with correlated variables (wavelengths or wave numbers), and project them into latent variables, which are in turn used for regression. 許多多變量資料集(例如：化學計量學中的光譜資料)其變量彼此相關

# Specific algorithms to deal with categorical (discrete) variables have been developed for PLS. This kinds of algorithm are grouped under the name of PLS Discriminant Analysis (PLS-DA). 一言以敝之，能處理類別/離散變量的PLS算法

# In this tutorial we are going to work through a binary classification problem with PLS-DA. In doing so, we’ll have the opportunity to look at PLS cross-decomposition under a different light, which I hope will help broaden the understanding of the wider decomposition (or cross-decomposition) approach to deal with multicollinearity. 偏最小平方法的交叉分解可有效處理多變量量間的多重共線性


#### A quick recap of PLS
# This is a key concept: there is no way around some sort of decomposition when dealing with spectral data. All spectra, from the point of view of a machine learning model, suffer from multicollinearity problems. 光譜資料一定得分解 ，因為共線性問題 Multicollinearity in a spectrum means that the signal at one wavelength is (somewhat or highly) correlated with the signal at neighbouring wavelengths. This correlation makes the mathematical problem of solving a least square regression ill-posed. In practice this means you can’t simply run a linear least square regression on the raw spectra, as the algorithm will fail (Note you can however modify the linear regression to introduce a regularisation of the least square problem. An example is the Ridge regression, but there’s more). 最小平方法會失敗

# Unlike PCA, PLS is a cross-decomposition technique. It derives the principal components by maximising the covariance between the spectra and the response variable (PCA on the contrary is based on minimising the covariance between the different spectra, without looking at the response variable). Therefore PLS will ensure that the first few principal components have a good degree of correlation with the response variable. This fact alone is arguably the reason why PLS is generally very successful in building calibration models in spectroscopy: in one swift operation, PLS can accomplish both dimensionality reduction and good correlation with the response.

# Now, PLS has been originally developed for (and mostly lends itself to) regression problems. Other dimensionality reduction techniques – such as PCA or LDA – are often/always used for classification. While being a dimensionality reduction technique itself, PLS is not generally applied to classification problems in the same way. Here’s a few reasons for that.


# 1. PCA decomposition doesn’t require the knowledge of the response variable. It therefore lends itself to both regression and classification problems, the latter in supervised or unsupervised fashion alike.
# 2. LDA decomposition is specified only for categorical response variables. It is therefore naturally applied to (supervised) classification.
# 3. PLS decomposition is inextricably linked to the response variable which is usually a continuous variable. PLS predictions therefore belong to a continuum, which is another way of saying that PLS is naturally applicable to regressions.

# A variation of PLS to deal with categorical variables however has been developed. The technique – or the class of techniques – is called Partial Least Square – Discriminant Analysis, PLS-DA. This class of techniques has been developed to answer the question: “What happens if we use categorical (numerical) response variables as input to a PLS decomposition?


#### Combining PLS and Discriminant Analysis
# To give an answer to this question, let’s look at the general structure of a PLS-DA algorithm. We’ll work through a classification problem using NIR data in the next section.

# The logical structure of PLS regression is very simple:

# 1. Run a PLS decomposition where the response vector contains real numbers 反應變量為實數
# 2. Run a linear regression on the principal components (or latent variables) obtained in the previous step.

# In fact this structure is exactly the same as the one used in PCR, where principal component analysis is done in place of PLS cross-decomposition.

# Then, simply enough, PLS-DA is done with a modification of the previous steps as follows:

# 1. Run a PLS decomposition where the response vector contains integer numbers 反應變量為整數下執行PLS分解
# 2. Instead of using the components as input to a linear regression, we use them as input to a classification problem. 對分解出來的成份建構分類模型

# The fact is that there is no condition in the PLS algorithm that constraints the output to be integers, regardless of the response variable. PLS was developed for continuous regression problems, and this fact keeps being true even if the variables are categorical. Sure enough, if the model is good, the test result will be close to either 0 or 1, but won’t generally be a match. 算法無法限制輸出為整數值

# Therefore one has to decide the best way to cluster the non-integer outputs so that they can be correctly assigned to a specified class. As mentioned, this step requires the judgement of the user, as there may be many ways to accomplish it.

# I hope that gives you an idea of the issue. Following the example in the next section, we’ll give an option for discriminant analysis using thresholding. 閾值(eg. 0.5)設定

#### PLS-DA for binary classification of NIR spectra

# The data set for this tutorial is a bunch of NIR spectra from samples of milk powder. Milk powder and coconut milk powder were mixed in different proportions and NIR spectra were acquired. The data set is freely available for download in our Github repo. 奶粉樣本

# The data set contains 11 different classes, corresponding to samples going from 100% milk powder to 0% milk powder (that is 100% coconut milk powder) in decrements of 10%. 奶粉 vs. 耶子奶粉比例 For the sake of running a binary classification, we are going to discard all mixes except the 5th and the 6th, corresponding to the 60/40 and 50/50 ratio of milk and coconut milk powder respectively. 奶粉比例100%到0%的11個不同類別，只取60/40和50/50二元分類

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
from scipy.signal import savgol_filter
 
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.metrics import accuracy_score


# Load data into a Pandas dataframe
data = pd.read_csv('./Data/milk-powder.csv') # (220, 603)
data.labels.value_counts() # 20 sample for each class

# Extract fifth and sixth label in a new dataframe
binary_data = data[(data['labels'] == 5 ) | (data['labels'] == 6)] # (40, 603)

# Read data into a numpy array and apply simple smoothing
X_binary = savgol_filter(binary_data.values[:,2:], window_length = 15, polyorder = 3, deriv=0) # (40, 601)

# Read categorical variables
y_binary = binary_data["labels"].values
# Map variables to 0 and 1
y_binary = (y_binary == 6).astype('uint8')


# Define the PLS regression object
pls_binary = PLSRegression(n_components=2)
# Fit and transform the data
X_pls = pls_binary.fit_transform(X_binary, y_binary)[0] # return (x_scores (40, 2), y_scores (40,)), doing dimensionaly reduction (DR) by the PLS regression


# Let’s now plot the result, making sure we keep track of the original labels

# Define the labels for the plot legend
labplot = ["60/40 ratio", "50/50 ratio"]
# Scatter plot
unique = list(set(y_binary))
colors = [plt.cm.jet(float(i)/max(unique)) for i in unique]
 
with plt.style.context(('ggplot')):
    plt.figure(figsize=(12,10))
    for i, u in enumerate(unique):
        col = np.expand_dims(np.array(colors[i]), axis=0)
        xi = [X_pls[j,0] for j in range(len(X_pls[:,0])) if y_binary[j] == u]
        yi = [X_pls[j,1] for j in range(len(X_pls[:,1])) if y_binary[j] == u]
        plt.scatter(xi, yi, c=col, s=100, edgecolors='k',label=str(u))
 
    plt.xlabel('Latent Variable 1')
    plt.ylabel('Latent Variable 2')
    plt.legend(labplot,loc='lower left')
    plt.title('PLS cross-decomposition')
    plt.show()


#### Prediction model by thresholding
# So far we have only accomplished dimensionality reduction by PLS cross-decomposition. To produce an actual classification model, we need to obtain binary predictions. Let’s work through this problem using a simple test-train split.

# Test-train split
X_train, X_test, y_train, y_test = train_test_split(X_binary, y_binary, test_size=0.2, random_state=19)
# Define the PLS object
pls_binary = PLSRegression(n_components=2)
# Fit the training set
pls_binary.fit(X_train, y_train)
 
# Predictions: these won't generally be integer numbers
y_pred = pls_binary.predict(X_test)[:,0]
# "Force" binary prediction by thresholding
binary_prediction = (pls_binary.predict(X_test)[:,0] > 0.5).astype('uint8')
print(binary_prediction, y_test)

# To be more thorough however, let’s estimate the model quality by cross-validation. We are going to borrow from our tutorial on cross-validation strategies which you can refer to for details on the implementation of the K-Fold cross-validation.

def pls_da(X_train, y_train, X_test):
    
    # Define the PLS object for binary classification
    plsda = PLSRegression(n_components=2)
    
    # Fit the training set
    plsda.fit(X_train, y_train)
    
    # Binary prediction on the test set, done with thresholding
    binary_prediction = (pls_binary.predict(X_test)[:,0] > 0.5).astype('uint8')
    
    return binary_prediction


# We have incorporated the thresholding step into the function, so to have binary predictions directly in output.

# Next, we are going to define a K-Fold cross-validation with 10 splits and iterate through it. At each step we calculate the accuracy score.

accuracy = []
cval = KFold(n_splits=10, shuffle=True, random_state=19)
for train, test in cval.split(X_binary): # <generator object _BaseKFold.split at 0x7ff0c0b93750> is produced by cval.split()
    
    y_pred = pls_da(X_binary[train,:], y_binary[train], X_binary[test,:])
    
    accuracy.append(accuracy_score(y_binary[test], y_pred))
 
print("Average accuracy on 10 splits: ", np.array(accuracy).mean())

accuracy

# This tells us that most of the classification runs give a perfect prediction (accuracy=1) while the accuracy drops in two cases. By looking at the scatter plot above, we may guess that the lower accuracy corresponds to the points near the centre of the plot being included in the test set. These cases are more ambiguous and a simple threshold classification is not able to accurately discriminate between them.

# As a side note, if you go back to the pre-processing step and set “deriv=1” the cross-decomposition achieves a better clustering and the classification accuracy will become 1 in all cases.


#### References:
# 1. https://www.mfitzp.com/tutorials/partial-least-squares-discriminant-analysis-plsda/
# 2. https://nirpyresearch.com/pls-discriminant-analysis-binary-classification-python/
# 3. https://data-farmers.github.io/2019-06-14-Partial-Least-Squares-Discriminant-Analysis/