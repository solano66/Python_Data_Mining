'''
Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the IDS, CICD, and CISD of NTUB (臺北商業大學資訊與決策科學研究所暨智能控制與決策研究室教授兼校務永續發展中心主任); at the ME Dept. and CAIDS of MCUT (2020年借調至明志科技大學機械工程系擔任特聘教授兼人工智慧暨資料科學研究中心主任兩年); the CSQ (2019年起任中華民國品質學會大數據品質應用委員會主任委員); the DSBA (2013年創立臺灣資料科學與商業應用協會); and the CARS (2012年創立中華R軟體學會) 
Notes: This code is provided without warranty.
'''

# Import required libraries
import numpy as np
import pandas as pd

# Set random seed
np.random.seed(42)

# Define total number of products
number_of_products = 10

# Create data dictionary
data = {'product_id':np.arange(1, number_of_products+1).tolist(),
       'measure':np.round(np.random.normal(loc=10, scale=0.5, size=number_of_products),3)} # np.round()

# Transform dictionary into a data frame
df = pd.DataFrame(data)

# Store the real mean in a separate variable
real_mean = round(df['measure'].mean(),3) # round()

# View data frame
df

### Simple Random Sampling
# Obtain simple random sample
simple_random_sample = df.sample(n=4).sort_values(by='product_id')

# Save the sample mean in a separate variable
simple_random_mean = round(simple_random_sample['measure'].mean(),3)

# View sampled data frame
simple_random_sample

### Systematic Sampling
# Define systematic sampling function
def systematic_sampling(df, step):
    
    indexes = np.arange(0,len(df),step=step)
    systematic_sample = df.iloc[indexes]
    return systematic_sample
    
# Obtain a systematic sample and save it in a new variable
systematic_sample = systematic_sampling(df, 3)

# Save the sample mean in a separate variable
systematic_mean = round(systematic_sample['measure'].mean(),3)

# View sampled data frame
systematic_sample

### Cluster Sampling
def cluster_sampling(df, number_of_clusters):
    
    try:
        # Divide the units into cluster of equal size
        df['cluster_id'] = np.repeat([range(1,number_of_clusters+1)],len(df)/number_of_clusters)

        # Create an empty list
        indexes = []

        # Append the indexes from the clusters that meet the criteria
        # For this formula, clusters id must be an even number
        for i in range(0,len(df)):
            if df['cluster_id'].iloc[i]%2 == 0:
                indexes.append(i)
        cluster_sample = df.iloc[indexes]
        return(cluster_sample)
    
    except:
        print("The population cannot be divided into clusters of equal size!")
        
# Obtain a cluster sample and save it in a new variable
cluster_sample = cluster_sampling(df, 5)

# Save the sample mean in a separate variable
cluster_mean = round(cluster_sample['measure'].mean(),3)

# View sampled data frame
cluster_sample

### Stratified Random Sampling
# Create data dictionary
data = {'product_id':np.arange(1, number_of_products+1).tolist(),
       'product_strata':np.repeat([1,2], number_of_products/2).tolist(),
       'measure':np.round(np.random.normal(loc=10, scale=0.5, size=number_of_products),3)}

# Transform dictionary into a data frame
df = pd.DataFrame(data)

# View data frame
df


# Import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit

# Set the split criteria
split = StratifiedShuffleSplit(n_splits=1, test_size=4)

# Perform data frame split
for x, y in split.split(df, df['product_strata']):
    stratified_random_sample = df.iloc[y].sort_values(by='product_id')

# View sampled data frame
stratified_random_sample

# Obtain the sample mean for each group
stratified_random_sample.groupby('product_strata').mean().drop(['product_id'],axis=1)

### Measure Mean Comparison per Sampling Method
# Create a dictionary with the mean outcomes for each sampling method and the real mean
outcomes = {'sample_mean':[simple_random_mean,systematic_mean,cluster_mean],
           'real_mean':real_mean}

# Transform dictionary into a data frame
outcomes = pd.DataFrame(outcomes, index=['Simple Random Sampling','Systematic Sampling','Cluster Sampling'])

# Add a value corresponding to the absolute error
outcomes['abs_error'] = abs(outcomes['real_mean'] - outcomes['sample_mean'])

# Sort data frame by absolute error
outcomes.sort_values(by='abs_error')

### Concluding Thoughts
# convenience sampling selection bias
# systematic sampling selection bias
# environmental bias
# non-response bias

### Probability Sampling with Python
# https://towardsdatascience.com/probability-sampling-with-python-8c977ad78664