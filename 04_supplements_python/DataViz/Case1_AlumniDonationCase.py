### Case 1: Exploratory Data Analysis: Alumni Donation Case ####
import pandas as pd

don = pd.read_csv('./data/contribution.csv', index_col=None, header=0)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

don.info(null_counts=True, verbose=True) # https://stackoverflow.com/questions/27637281/what-are-python-pandas-equivalents-for-r-functions-like-str-summary-and-he

don.head()

don.describe(include='all')

don['Class Year'].value_counts()
don['Class Year'].value_counts().plot.bar() # 長條圖barplot by Pandas Series簡便繪圖函數

# don.groupby(['Class Year'])['Gender'].count()
freq = don.groupby(['Class Year']).size()
import seaborn as sns
sns.barplot(x=freq.index, y=freq, data=freq) # 語法很像R語言繪圖函數

import numpy as np
import matplotlib.pyplot as plt

height = don['Class Year'].value_counts(ascending = True)
# bars = np.sort(don['Class Year'].value_counts().index)
# x_pos = np.arange(len(bars))
# pandas - change df.index from float64 to unicode or string (https://stackoverflow.com/questions/35368645/pandas-change-df-index-from-float64-to-unicode-or-string)
# height.index = height.index.map(str)

# 先高階繪圖(high-level plotting)
plt.bar(height.index.map(str), height)
# 再做低階繪圖(low-level plotting)(畫龍點睛，為您的圖形增添色彩！)
#plt.xticks(x_pos, height.index)
plt.xlabel('Class Year')
plt.ylabel('Frequency')
plt.title('Frequency Distribution of Class Year')
plt.show()

don['TGiving'] = don.FY00Giving + don.FY01Giving + don.FY02Giving + don.FY03Giving + don.FY04Giving
don.info()

don.TGiving.describe()
don.TGiving.mean()
don.TGiving.std()
don.TGiving.quantile(np.arange(0.0,1.05,0.05))
don.TGiving.quantile(np.arange(.95,1.0,0.01))

# matplotlib.pyplot的直方圖
plt.hist(don.TGiving)
plt.show();

plt.hist(don.TGiving[(don.TGiving != 0) & (don.TGiving <= 1000)])
plt.show();

# 另一個了解量化變量分布的盒鬚圖
plt.boxplot(don.TGiving, '*')
plt.show()

plt.boxplot(don.TGiving, notch=True, sym='') # Enter an empty string ('') if you don't want to show fliers.
plt.show()

plt.boxplot(don.TGiving, 0, '', vert=False)
plt.xlabel('Total Contribution')
plt.show()

ddd = don[don.TGiving>=30000]
ddd1 = ddd[['Gender', 'Class Year', 'Marital Status', 'Major', 'Next Degree', 'TGiving']]
ddd1 
ddd1.sort_values(by='TGiving', ascending=False)

# 多變量視覺化(類別變量 + 量化變量)
# It is important to know who is contributing, so box plots of total 5-year donation for the class year, gender, marital status, and attendance at a event
import seaborn as sns
sns.boxplot(x="Class Year", y="TGiving" ,data=don, showfliers=False, width=0.5)
sns.boxplot(x="Class Year", y="TGiving" ,data=don, width=0.5) # 有outliers

sns.boxplot(x="Gender", y="TGiving" ,data=don, showfliers=False, width=0.5)
sns.boxplot(x="Marital Status", y="TGiving" ,data=don, showfliers=False, width=0.5)
sns.boxplot(x="AttendenceEvent", y="TGiving" ,data=don, showfliers=False, width=0.5)

# Boxplots of total giving against the alumni's major and second degree.
t4 = don.groupby('Major')['TGiving'].mean()
len(t4)
t5 = don['Major'].value_counts()
len(t5)
t6 = pd.concat([t4, t5], axis=1, sort = False)
t7 = t6[t6.iloc[:,1]>10]
t7 = t7.sort_values(by=t7.columns[0], ascending=False)
sns.barplot(x=t7.columns[0], y=t7.index, color='black', data=t7).set(xlabel='Mean TGiving', title='TGiving against Major')

t4 = don.groupby('Next Degree')['TGiving'].mean()
len(t4)
t5 = don['Next Degree'].value_counts()
len(t5)
t6 = pd.concat([t4, t5], axis=1, sort = False)
t7 = t6[t6.iloc[:,1]>10]
t7 = t7.sort_values(by=t7.columns[0], ascending=False)
sns.barplot(x=t7.columns[0], y=t7.index, color='black', data=t7).set(xlabel='Mean TGiving', title='TGiving against Next Degree')

# The distribution of 5-year giving among alumni who gave $1 - $1000, stratified according to year of graduation.
ordered_years = np.sort(don['Class Year'].value_counts().index)
g = sns.FacetGrid(don[(don.TGiving != 0) & (don.TGiving <= 1000)], col="Class Year", col_order=ordered_years, col_wrap=3)
g.map(sns.distplot, "TGiving", hist=False, rug=False, color="black")

# Calculating the total 5-year donations for the five graduation cohorts
t11 = don.groupby('Class Year')['TGiving'].sum()
t11
sns.barplot(x=t11.index, y=t11.values).set(xlabel='Class Year', ylabel="Total Donation")

# Annual contributions (2000-2004) of the five graduation classes
sns.barplot(x=don.groupby('Class Year')['FY04Giving'].sum().index, y=don.groupby('Class Year')['FY04Giving'].sum().values, color='black').set(ylim=(0, 225000))
sns.barplot(x=don.groupby('Class Year')['FY03Giving'].sum().index, y=don.groupby('Class Year')['FY03Giving'].sum().values, color='black').set(ylim=(0, 225000))
sns.barplot(x=don.groupby('Class Year')['FY02Giving'].sum().index, y=don.groupby('Class Year')['FY02Giving'].sum().values, color='black').set(ylim=(0, 225000))
sns.barplot(x=don.groupby('Class Year')['FY01Giving'].sum().index, y=don.groupby('Class Year')['FY01Giving'].sum().values, color='black').set(ylim=(0, 225000))
sns.barplot(x=don.groupby('Class Year')['FY00Giving'].sum().index, y=don.groupby('Class Year')['FY00Giving'].sum().values, color='black').set(ylim=(0, 225000))

# Computing and analyzing the numbers and proportions of individuals who contributed (分析捐款人數與比例)
don.TGiving.describe()
np.sort(don.TGiving)[1:425]
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.cut.html
don["TGivingIND"] = pd.cut(don.TGiving, [-1,0.5,10000000], labels=False)
#don.TGivingIND.value_counts()
#1    808
#0    422
don.TGivingIND.mean()
t5 = pd.crosstab(index=don["TGivingIND"], columns=don["Class Year"], margins=False).reset_index()
t5 = pd.melt(t5, id_vars=['TGivingIND'], value_vars=[1957,1967,1977,1987,1997], var_name='Class Year', value_name='count')
sns.barplot(x='Class Year', y="count", hue="TGivingIND", data=t5)

from statsmodels.graphics.mosaicplot import mosaic
mosaic(don, ['Class Year', 'TGivingIND'], gap=0.01)
t50 = don.groupby('Class Year')['TGivingIND'].mean()
t50
sns.barplot(x=t50.index, y=t50.values, color='black').set(xlabel='Class Year', title='Donation Proportion')

don["FY04GivingIND"] = pd.cut(don.FY04Giving, [-1,0.5,10000000], labels=False)
#don["FY04GivingIND"].value_counts()
#0    723
#1    507
t51 = don.groupby('Class Year')['FY04GivingIND'].mean()
t51
sns.barplot(x=t51.index, y=t51.values, color='black').set(xlabel='Class Year', title='Donation Proportion of FY04Giving')

# Exploring the relationship between the alumni contributions among the 5 years (五年捐款金額關係)
#602
Data = pd.DataFrame({'FY04Giving':don.FY04Giving, 'FY03Giving':don.FY03Giving, 'FY02Giving':don.FY02Giving, 'FY01Giving':don.FY01Giving, 'FY00Giving':don.FY00Giving})
correlation = Data.corr()
correlation

g = sns.PairGrid(Data)
g.map(plt.scatter);
#g.map_diag(plt.hist)
#g.map_offdiag(plt.scatter);

f, ax = plt.subplots(figsize=(10, 8))
ax = sns.heatmap(
    correlation, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(220, 10, as_cmap=True),
    square=True, 
    linewidths=.5, 
    cbar_kws={"shrink": .8},
    annot=True, annot_kws={"size": 14}
)
ax.set_yticklabels(ax.get_yticklabels(), rotation = 0)

mosaic(don, ['Gender', 'TGivingIND'], gap=0.01)
mosaic(don, ['Marital Status', 'TGivingIND'], gap=0.01)
mosaic(don, ['AttendenceEvent', 'TGivingIND'], gap=0.01)

#t2 = don.groupby(['AttendenceEvent','Marital Status'])['TGivingIND'].value_counts()
t2 = pd.crosstab([don['AttendenceEvent'],don['Marital Status']],don['TGivingIND'])
t2

fig, (ax, ax2) = plt.subplots(figsize=(15, 8),nrows=1, ncols=2)
mosaic(don[don.AttendenceEvent==0], ['Marital Status','TGivingIND'], gap=0.01, ax=ax)
mosaic(don[don.AttendenceEvent==1], ['Marital Status','TGivingIND'], gap=0.01, ax=ax2)
plt.show()
