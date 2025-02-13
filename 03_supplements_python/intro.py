# # Introduction
# ## Before You Start
#### Initial Data Analysis
#%%	

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns # Multivariate Plotting
import statsmodels.formula.api as smf


#%%	

import faraway.datasets.pima # pip install faraway
pima = faraway.datasets.pima.load()
pima.head()


#%%	

print(faraway.datasets.pima.DESCR)

# Diabetes survey on Pima Indians

# Description

# The National Institute of Diabetes and Digestive and Kidney Diseases conducted a study on 768 adult female Pima Indians living near Phoenix.

# Usage

# data(pima)
# Format

# The dataset contains the following variables

# pregnant
# Number of times pregnant 懷孕了幾次

# glucose
# Plasma glucose concentration at 2 hours in an oral glucose tolerance test 血液中葡萄糖濃度

# diastolic
# Diastolic blood pressure (mm Hg) 舒張壓 mm Hg

# triceps
# Triceps skin fold thickness (mm) 三頭肌皮摺厚度

# insulin
# 2-Hour serum insulin (mu U/ml) 血清胰島素濃度

# bmi
# Body mass index (weight in kg/(height in metres squared)) BMI值

# diabetes
# Diabetes pedigree function 糖尿病函數，這個函數使用了家族糖尿病史來導出個人得糖尿病的風險值

# age
# Age (years)

# test
# test whether the patient shows signs of diabetes (coded 0 if negative, 1 if positive) 0 無糖尿病，1 有糖尿病

# Source

# The data may be obtained from UCI Repository of machine learning databases at http://archive.ics.uci.edu/ml/

#%%	

pima_summ = pima.describe().round(1)


#%%	

pima['diastolic'].sort_values().head()


#%%
# 35 missing values in diastolic blood pressure
np.sum(pima['diastolic'] == 0)

#%%
# Originally no missing value !
pima.isnull().sum()

#%%

# Replace all 0's with np.nan and check it again
pima.replace({'diastolic' : 0, 'triceps' : 0, 'insulin' : 0, 
    'glucose' : 0, 'bmi' : 0}, np.nan, inplace=True)

pima.isnull().sum()

#%%
# Data types are all integers and floats
pima.dtypes

#%%

pima['test'] = pima['test'].astype('category')
pima['test'] = pima['test'].cat.rename_categories(
    ['Negative','Positive'])
pima['test'].value_counts()


#%%	

sns.displot(pima.diastolic.dropna(), kde=True)


#%%	

pimad = pima.diastolic.dropna().sort_values()
# By default, the plot aggregates over multiple y values at each value of x and shows an estimate of the central tendency and a confidence interval for that estimate.
sns.lineplot(x=range(0, len(pimad)), y=pimad)


#%%	

sns.scatterplot(x='diastolic',y='diabetes',data=pima, s=20) # s: size


#%%	

sns.boxplot(x="test", y="diabetes", data=pima)


#%%	
# superimposed plot 疊加圖
sns.scatterplot(x="diastolic", y="diabetes", data=pima, 
    style="test", alpha=0.3)


#%%	

sns.relplot(x="diastolic", y="diabetes", data=pima, col="test")

#%%	
# ## When to Use Linear Modeling
# ## History

import faraway.datasets.manilius
manilius = faraway.datasets.manilius.load()
manilius.head()


#%%	

moon3 = manilius.groupby('group').sum()
moon3


#%%

moon3['intercept'] = [9]*3
np.linalg.solve(moon3[['intercept','sinang','cosang']],
    moon3['arc'])


#%%	

mod = smf.ols('arc ~ sinang + cosang', manilius).fit()
mod.params


#%%	

import faraway.datasets.families
families = faraway.datasets.families.load()
sns.scatterplot(x='midparentHeight', y='childHeight',
    data=families, s=20)


#%%	

mod = smf.ols('childHeight ~ midparentHeight', families).fit()
mod.params


#%%	

cor = sp.stats.pearsonr(families['childHeight'],
    families['midparentHeight'])[0]
sdy = np.std(families['childHeight'])
sdx = np.std(families['midparentHeight'])
beta = cor*sdy/sdx
alpha = np.mean(families['childHeight']) - \
    beta*np.mean(families['midparentHeight'])
np.round([alpha,beta],2)


#%%	

beta1 = sdy/sdx
alpha1 = np.mean(families['childHeight']) - \
    beta1*np.mean(families['midparentHeight'])


#%%	

sns.lmplot(x='midparentHeight', y='childHeight', data=families, 
    ci=None, scatter_kws={'s':2})
xr = np.array([64,76])
plt.plot(xr, alpha1 + xr*beta1,'--')

#%%
# ## Exercises

# ## Packages Used

import sys
import matplotlib
import statsmodels as sm
import seaborn as sns
print("Python version:{}".format(sys.version))
print("matplotlib version: {}".format(matplotlib.__version__))
print("pandas version: {}".format(pd.__version__))
print("numpy version: {}".format(np.__version__))
print("statsmodels version: {}".format(sm.__version__))
print("seaborn version: {}".format(sns.__version__))

    
# %%
