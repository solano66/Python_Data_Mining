
import pandas as pd
# creat a data frame with duplicated items from a dict
data = {'HouseType':['Semi','Single','Row','Single','Apartment','Apartment','Row'], 'HouseTypeNo':[1,2,3,2,4,4,3]}
hous = pd.DataFrame(data)
hous.info()

# create a lookup table from above data frame to get the numbers to fill the large table
lookup = hous.drop_duplicates()

# create a toy large table with a 'HouseType' column, but no 'HouseTypeNo' column (yet)
from random import choices
hous['HouseType'].unique()
largetable = pd.DataFrame({'HouseType':choices(hous['HouseType'].unique(), k=1000)})

# 1. using pandas DataFrame merge() 
base1 = pd.merge(lookup, largetable, on='HouseType')
base1.head() # order is not same as largetable !

# 2. using pandas Series map() and a dict
housenames = dict(zip(hous['HouseType'].unique(), range(1, len(hous['HouseType'].unique())+1)))
housenames['Row']
base2 = pd.DataFrame({'HouseType':list(largetable['HouseType']), 'HouseTypeNo': largetable['HouseType'].map(housenames)})
# base2['HouseTypeNo'] = largetable['HouseType'].map(housenames) 
base2.head() # order is same as largetable !

# 3. using join
join1 = largetable.join(lookup.set_index('HouseType'), on='HouseType')
join1.head(25) # order is same as the largetable

# 4. using the pandasql package
import pandasql as ps # conda install pandasql --y
q1 = "SELECT largetable.HouseType, lookup.HouseTypeNo FROM largetable INNER JOIN lookup ON largetable.HouseType = lookup.HouseType "
ps.sqldf(q1, locals())

# If it's possible that some house types in largetable do not exist in lookup then a left join would be used:
q2 ="select * from largetable left join lookup using (HouseType)"
ps.sqldf(q2, locals())




