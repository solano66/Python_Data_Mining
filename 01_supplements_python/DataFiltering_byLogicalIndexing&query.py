'''
Collated by Ching-Shih Tsou 鄒慶士 博士 (Ph.D.) Distinguished Prof. at the Department of Mechanical Engineering/Director at the Center of Artificial Intelligence & Data Science (機械工程系特聘教授兼人工智慧暨資料科學研究中心主任), MCUT (明志科技大學); Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), NTUB (國立臺北商業大學)
Notes: This code is provided without warranty.
'''

# load pandas
import pandas as pd
data_url = 'http://bit.ly/2cLzoxH'
# read data from url as pandas dataframe
gapminder = pd.read_csv(data_url)

### How to Select Rows of Pandas Dataframe Based on a Single Value of a Column?
# does year equals to 2002?
# is_2002 is a boolean variable with True or False in it
is_2002 = gapminder['year']==2002
print(is_2002.head())

# filter rows for year 2002 using  the boolean variable
gapminder_2002 = gapminder[is_2002]
print(gapminder_2002.shape)

print(gapminder_2002.head())

# filter rows for year 2002 using  the boolean expression
gapminder_2002 = gapminder[gapminder['year']==2002]
print(gapminder_2002.shape) # same as above

### How To Filter rows using Pandas chaining?
# filter rows for year 2002 using  the boolean expression
gapminder_2002 = gapminder[gapminder.year.eq(2002)]
print(gapminder_2002.shape) # same as above

### How to Select Rows of Pandas Dataframe Whose Column Value Does NOT Equal a Specific Value?
# filter rows for year does not equal to 2002
gapminder_not_2002 = gapminder[gapminder.year != 2002]
gapminder_not_2002 = gapminder[gapminder['year']!=2002]
gapminder_not_2002.shape

### How to Select Rows of Pandas Dataframe Whose Column Value is NOT NA/NAN?
# filter out rows ina . dataframe with column year values NA/NAN
gapminder_no_NA = gapminder[gapminder.year.notnull()]

### How to Select Rows of Pandas Dataframe Based on a list?
#To select rows whose column value is in list 
years = [1952, 2007]
gapminder.year.isin(years)

gapminder_years= gapminder[gapminder.year.isin(years)]
gapminder_years.shape

gapminder_years.year.unique()

### How to Select Rows of Pandas Dataframe Based on Values NOT in a list?
continents = ['Asia','Africa', 'Americas', 'Europe']
gapminder_Ocean = gapminder[~gapminder.continent.isin(continents)]
gapminder_Ocean.shape 

### How to Select Rows of Pandas Dataframe using Multiple Conditions?
gapminder[~gapminder.continent.isin(continents) & 
           gapminder.year.isin(years)]

### How to Filter Rows of Pandas Dataframe with Query function?
# filter rows with Pandas query
gapminder.query('country=="United States"').head()
gapminder.query('year==1952').head()
gapminder.query('country=="United States" & year > 1996')
gapminder.query('country in ["United States", "United Kingdom"] & year > 2000')

# Using backtick quoting for more than only spaces
df = pd.DataFrame({'A': range(1, 6),
                   'B': range(10, 0, -2),
                   'C C': range(10, 5, -1)})
df.query('B == `C C`')
#    A   B  C C
# 0  1  10   10

### References:
# 1. How To Filter Pandas Dataframe By Values of Column? (https://cmdlinetips.com/2018/02/how-to-subset-pandas-dataframe-based-on-values-of-a-column/)
# 2. How to Filter Rows of Pandas Dataframe with Query function? (https://cmdlinetips.com/2019/07/how-to-select-rows-of-pandas-dataframe-with-query-function/)
