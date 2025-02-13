#!/usr/bin/env python
# coding: utf-8

# # Just pandas things
# 
# It's possible that Python wouldn't have become [the lingua franca of data science if it wasn't for pandas](https://stackoverflow.blog/2017/09/14/python-growing-quickly/). The package's exponential growth on Stack Overflow means two things:
# 1. It's getting increasingly popular.
# 2. It can be frustrating to use sometimes (hence the high number of questions).
# 
# This repo contains a few peculiar things I've learned about pandas that have made my life easier and my code faster. This post isn't a friendly tutorial for beginners, but a friendly introduction to pandas weirdness.
# 
# To demonstrate the use of pandas, we'll be using the interview reviews scraped from Glassdoor in 2019. The data is stored in the folder `data` under the name `interviews.csv`.
# 
# I'll continue updating this repo as I have more time. As I'm still learning pandas quirks, feedback is much appreciated!
# 
# ![](https://i.pinimg.com/originals/9e/7c/78/9e7c7816c30327890dc94ba16e5dac1b.jpg)

# In[1]:


import pandas as pd
df = pd.read_csv("interviews.csv")

print(df.shape)
df.head()

df.info()
df.Review

#### 1. pandas is column-major
# Pandas is built around `DataFrame`, a concept inspired by R's Data Frame, which is, in turn, similar to tables in relational databases. A `DataFrame` is a two-dimensional table with rows and columns.
# 
# One important thing to know about pandas is that it's column-major, which explains many of its quirks.
# 
# Column-major means consecutive elements in a column are stored next to each other in memory. Row-major means the same but for elements in a row. Because modern computers process sequential data more efficiently than nonsequential data, if a table is row-major, accessing its rows will be much faster than accessing its columns.
# 
# In NumPy, major order can be specified. When a `ndarray` is created, it's row-major by default if you don't specify the order.
# 
# Like R's Data Frame, pandas' `DataFrame` is column-major. People coming to pandas from NumPy tend to treat `DataFrame` the way they would `ndarray`, e.g. trying to access data by rows, and find `DataFrame` slow.
# 
# **Note**: A column in a `DataFrame` is a `Series`. You can think of a `DataFrame` as a bunch of `Series` being stored next to each other in memory.
# 
# **For our dataset, accessing a row takes about 50x longer than accessing a column in our `DataFrame`.**

# In[2]:


# Get the column `date`, 1000 loops
get_ipython().run_line_magic('timeit', '-n1000 df["Date"]')

# Get the first row, 1000 loops
get_ipython().run_line_magic('timeit', '-n1000 df.iloc[0]')


# ### 1.1 Iterating over rows
# #### 1.1.1 `.apply()`
# pandas documentation has [a warning box](https://pandas.pydata.org/pandas-docs/stable/user_guide/basics.html#iteration) that basically tells you not to iterate over rows because it's slow.
# 
# Before iterating over rows, think about what you want to do with each row, pack that into a function and use methods like `.apply()` to apply the function to all rows.
# 
# For example, to scale the "Experience" column by the number of "Upvotes" each review has, one way is to iterate over rows and multiple the "Upvotes" value by the "Experience" value of that row. But you can also use `.apply()` with a `lambda` function.

# In[3]:


get_ipython().run_line_magic('timeit', '-n1 df.apply(lambda x: x["Experience"] * x["Upvotes"], axis=1)')


# #### 1.1.2 `.iterrows()` and `.itertuples()`
# If you really want to iterate over rows, one naive way is to use `.iterrows()`. It returns a generator that generates row by row and it's very slow.

# In[4]:


get_ipython().run_line_magic('timeit', '-n1 [row for index, row in df.iterrows()]')


# In[5]:


# This is what a row looks like as a pandas object
for index, row in df.iterrows():
    print(row)
    break


# `.itertuples()` returns rows in the `namedtuple` format. It still lets you access each row and it's about 40x faster than `.iterrows()`.

# In[6]:


get_ipython().run_line_magic('timeit', '-n1 [row for row in df.itertuples()]')


# In[7]:


# This is what a row looks like as a namedtuple.
for row in df.itertuples():
    print(row)
    break


# #### 1.1.3 Converting DataFrame to row-major order
# If you need to do a lot of row operations, you might want to convert your `DataFrame` to a NumPy's row-major `ndarray`, then iterating through the rows.

# In[8]:


# Now, iterating through our DataFrame is 100x faster.
get_ipython().run_line_magic('timeit', '-n1 df_np = df.to_numpy(); rows = [row for row in df_np]')


# Accessing a row or a column of our `ndarray` takes nanoseconds instead of microseconds.

# In[9]:


df_np = df.to_numpy()
get_ipython().run_line_magic('timeit', '-n1000 df_np[0]')
get_ipython().run_line_magic('timeit', '-n1000 df_np[:,0]')


# ### 1.2. Ordering slicing operations
# Because pandas is column-major, if you want to do multiple slicing operations, always do the column-based slicing operations first.
# 
# For example, if you want to get the review from the first row of the data, there are two slicing operations:
# - get row (row-based operation)
# - get review (column-based operation)
# 
# Get row -> get review is 25x slower than get review -> get row.
# 
# **Note**: You can also just use `df.loc[0, "Review"]` to calculate the memory address to retrieve the item. Its performance is comparable to get review then get row.

# In[10]:


get_ipython().run_line_magic('timeit', '-n1000 df["Review"][0]')
get_ipython().run_line_magic('timeit', '-n1000 df.iloc[0]["Review"]')
get_ipython().run_line_magic('timeit', '-n1000 df.loc[0, "Review"]')


# ## 2. SettingWithCopyWarning
# Sometimes, when you try to assign values to a subset of data in a DataFrame, you get `SettingWithCopyWarning`. Don't ignore the warning because it means sometimes, the assignment works (example 1), but sometimes, it doesn't (example 2).

# In[11]:


# Example 1: Changing the review of the first row.
df["Review"][0] = "I like Orange better."
# Even though with the warning, the assignment works. The review is updated.
df.head(1)


# In[12]:


# Example 2: Changing the company name Apple to Orange.
df[df["Company"] == "Apple"]["Company"] = "Orange"
# With the warning, the assignment doesn't work. The company name is still Apple.
df.head(1)


# ### 2.1. `View` vs. `Copy`
# To understand this weird behavior, we need to understand two concepts in pandas: `View` vs. `Copy`.
# - `View` is the actual `DataFrame` you want to work with.
# - `Copy` is a copy of that actual `DataFrame`, which will be thrown away as soon as the operation is done.
# 
# So if you try to do an assignment on a `Copy`, the assignment won't work. 
# 
# `SettingWithCopyWarning` doesn't mean you're making changes to a `Copy`. It means that the thing you're making changes to might be a `Copy` or a `View`, and pandas can't tell you.
# 
# The ambiguity happens because of `__getitem__` operation.
# `__getitem__` sometimes returns a `Copy`, sometimes a `View`, and pandas makes no guarantee.

# In[13]:


# df["Review"][0] = "I like Orange better."
# can be understood as
# `df.__getitem__("Review").__setitem__(0, "I like Orange better.")`


# In[14]:


# df[df["Company"] == "Apple"]["Company"] = "Orange"
# can be understood as
# df.__getitem__(where df["Company"] == "Apple").__setitem__("Company", "Orange")


# ### 2.2 Solutions
# #### 2.2.1 Combine all chained operations into one single operation
# To avoid `__getitem__` ambiguity, you can combine all your operations into one single operation.
# `.loc[]` is usually great for that.

# In[15]:


# Changing the review of the first row.
df.loc[0, "Review"] = "Orange is love. Orange is life."
df.head()


# In[16]:


# Changing the company name Apple to Orange.
df.loc[df["Company"] == "Apple", "Company"] = "Orange"
df.head()


# #### 2.2.2 Raise an error
# I believe `SettingWithCopyWarning` should be an Exception instead of a warning. You can change this warning into an exception with pandas' magic `set_option()`.

# In[17]:


pd.set_option("mode.chained_assignment", "raise")
# Running this will show you an Exception
# df["Review"][0] = "I like Orange better."


# ## 3. Indexing and slicing
# 
# ### 3.1 `.iloc[]`: selecting rows based on integer indices
# `.iloc[]` lets you select rows by integer indices.

# In[18]:


# Accessing the third row of a `DataFrame`.
df.iloc[3]


# Slicing with `.iloc[]` is similar to slicing in Python. If you want a refresh on how slicing in Python works, see [Python-is-cool](https://github.com/chiphuyen/python-is-cool).

# In[19]:


# Selecting the last 6 rows
df.iloc[-6:]


# In[20]:


# Selecting 1 from every 2 rows in the last 6 rows
df.iloc[-6::2]


# ### 3.2 `.loc[]`: selecting rows by labels or boolean masks
# `.loc[]` lets you select rows based on one of the two things:
# - boolean masks
# - labels
# 
# #### 3.2.1 Selecting rows by boolean masks
# If you want to select all the rows where candidates declined offer, you can do it with two steps:
# 1. Create a boolean mask on whether the "Offer" column equals to "Declined offer"
# 2. Use that mask to select rows

# In[21]:


df.loc[df["Offer"] == "Declined offer"]
# This is equivalent to:
# mask = df["Offer"] == "Declined offer"
# df.loc[mask]


# #### 3.2.2 Selecting rows by labels
# ##### 3.2.2.1 Creating labels
# Currently, our `DataFrame` has no labels yet. To create labels, use `.set_index()`.
# 
# 1. Labels can be integers or strings
# 2. A DataFrame can have multiple labels

# In[22]:


# Adding label "Hardware" if the company name is "Orange", "Dell", "IDM", or "Siemens".
# "Orange" because we changed "Apple" to "Orange" above.
# Adding label "Software" otherwise.

def company_type(x):
    hardware_companies = set(["Orange", "Dell", "IBM", "Siemens"])
    return "Hardware" if x["Company"] in hardware_companies else "Software"
df["Type"] = df.apply(lambda x: company_type(x), axis=1)

# Setting "Type" to be labels. We call ""
df = df.set_index("Type")
df
# Label columns aren't considered part of the DataFrame's content.
# After adding labels to your DataFrame, it still has 10 columns, same as before.


# **Warning**: labels in `DataFrame` are stored as normal columns when you write the `DataFrame` to file using `.to_csv()`, and will need to be explicitly set after loading files, so if you send your CSV file to other people without explaining, they'll have no way of knowing which columns are labels. This might cause reproducibility issues. See [Stack Overflow answer](https://stackoverflow.com/questions/20109391/how-to-make-good-reproducible-pandas-examples).

# ##### 3.2.2.1 Selecting rows by labels 

# In[23]:


# Selecting rows with label "Hardware"
df.loc["Hardware"]


# In[24]:


# To drop a label, you need to use reset_index with drop=True
df.reset_index(drop=True, inplace=True)
df


# ### 3.3 Slicing Series
# Slicing pandas `Series` is similar to slicing in Python.

# In[25]:


series = df.Company
# The first 1000 companies, picking every 100th companies
series[:1000:100]


# ## 4. Accessors
# 
# ### 4.1 string accessor
# `.str` allows you to apply built-in string functions to all strings in a column (aka a pandas Series). These built-in functions come in handy when you want to do some basic string processing.

# In[26]:


# If you want to lowercase all the reviews in the `Reviews` column.
df["Review"].str.lower()


# In[27]:


# Or if you want to get the length of all the reviews
df.Review.str.len()


# `.str` can be very powerful if you use it with Regex. Imagine you want to get a sense of how long the interview process takes for each review. You notice that each review mentions how long it takes such as "the process took 4 weeks". So you use this heuristic:
# - a process is short if it takes days
# - a process is average is if it takes weeks
# - a process is long if it takes at least 4 weeks

# In[28]:


df.loc[df["Review"].str.contains("days"), "Process"] = "Short"
df.loc[df["Review"].str.contains("week"), "Process"] = "Average"
df.loc[df["Review"].str.contains("month|[4-9]+[^ ]* weeks|[1-9]\d{1,}[^ ]* weeks"), "Process"] = "Long"
df[~df.Process.isna()][["Review", "Process"]]


# We want to sanity check if `Process` corresponds to `Review`, but `Review` is cut off in the display above. To show wider columns, you can set `display.max_colwidth` to `100`.
# 
# **Note**: set_option has several great options you should check out.

# In[29]:


pd.set_option('display.max_colwidth', 100)
df[~df.Process.isna()][["Review", "Process"]]


# In[30]:


# To see the built-in functions available for `.str`, use this
pd.Series.str.__dict__.keys()


# ### 4.2 Other accessors
# pandas `Series` has 3 other accessors.
# - `.dt`: handles date formats
# - `.cat`: handles categorical data
# - `.sparse`: handles sparse matrices

# In[31]:


pd.Series._accessors


# ## 5. Data exploration
# When analyzing data, you might want to take a look at the data. pandas has some great built-in functions for that.
# 
# ### 5.1 `.head()`, `.tail()`, `.describe()`, `.info()`
# You're probably familiar with `.head()` and `.tail()` methods for showing the first/last rows of `DataFrame`. By default, 5 rows are shown, but you can specify the exact number.

# In[32]:


df.tail(8)


# In[33]:


# Generate statistics about numeric columns.
df.describe()

# Note:
# 1. `.describe()` ignores all non-numeric columns.
# 2. It doesn't take into account NaN values. So, the number shown in `count` below is the number of non-NaN entries.


# In[34]:


# Show non-null count and types of all columns
df.info()

# Note: pandas treats string type as object


# In[35]:


# You can also see how much space your DataFrame is taking up
import sys
df.apply(sys.getsizeof)


# ### 5.2 Count unique values
# You can get the number of unique values in a row (excluding NaN) with `nunique()`.

# In[36]:


# Get the number of unique companies in our data
df.Company.nunique()


# In[37]:


# You can also see how many reviews are for each company, sorted in a descending order.
df.Company.value_counts()


# ### 5.3 Plotting
# If you want to see the break down of process lengths for different companies, you can use `.plot()` with `.groupby()`.
# 
# **Note**: Plotting in pandas is both mind-boggling and mind-blowing. If you're not familiar, you might want to check out some tutorials, e.g. [this simple tutorial](https://realpython.com/pandas-plot-python/) or [this saiyan-level pandas plotting with seaborn](https://jakevdp.github.io/PythonDataScienceHandbook/04.14-visualization-with-seaborn.html).

# In[38]:


# Group the DataFrame by "Company" and "Process", count the number of elements,
# then unstack by "Process", then plot a bar chart
df.groupby(["Company", "Process"]).size().unstack(level=1).plot(kind="bar", figsize=(15, 8))


# ## 6. Common pitfalls
# pandas is great for most day-to-day data analysis. It's instrumental to my job and I'm grateful that the entire pandas community is actively developing it. However, I think some of pandas design decisions are a bit questionable.
# 
# Some of the common pandas pitfalls:
# ### 6.1 NaNs
# NaNs are stored as floats in pandas, so when an operation fails because of NaNs, it doesn't say that there's a NaN but because that operation doesn't exist for floats.
# 
# ### 6.2 Changes not Inplace 
# Most pandas operations aren't inplace by default, so if you make changes to your `DataFrame`, you need to assign the changes back to your DataFrame. You can make changes inplace by setting argument `inplace=True`.

# In[39]:


# "Process" column is still in df
df.drop(columns=["Process"])
df.columns


# In[40]:


# To make changes to df, set `inplace=True`
df.drop(columns=["Process"], inplace=True)
df.columns
# This is equivalent to
# df = df.drop(columns=["Process"])


# ### 6.3 Performance issues with very large datasets
# 
# ### 6.4 Reproducibility issues
# Especially with dumping and loading `DataFrame` to/from files. There are two main causes:
# 
# - Problem with labels (see the section about labels above).
# - [Weird rounding issues for floats](https://stackoverflow.com/questions/47368296/pandas-read-csv-file-with-float-values-results-in-weird-rounding-and-decimal-dig). 
# 
# ### 6.5 Not GPU compatible
# pandas can't take advantage of GPUs, so if your computations are on on GPUs and your feature engineering is on CPUs, it can become a time bottleneck to move data from CPUs to GPUs. If you want something like pandas but works on GPUs, check out dask and modin.

# Oh, and this is [the analysis I did based on this data](https://huyenchip.com/2019/08/21/glassdoor-interview-reviews-tech-hiring-cultures.html), in case you're interested!
