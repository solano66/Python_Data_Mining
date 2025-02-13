'''
Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the IDS and CADS (資訊與決策科學研究所暨資料科學應用研究中心), NTUB (國立臺北商業大學); the CARS(中華R軟體學會); and the DSBA(臺灣資料科學與商業應用協會)
Datasets: mtcars.csv
Notes: This code is provided without warranty.
'''

'''
Heat map
Choropleth
'''

import pandas as pd
mtcars = pd.read_csv('./data/mtcars.csv', index_col=0)

mtcars.dtypes
mtcars.describe(include='all')
dir(pd)
pd.get_option('display.max_columns') # In case python/IPython is running in a terminal and large_repr equals ‘truncate’ this can be set to 0 and pandas will auto-detect the width of the terminal and print a truncated object which fits the screen width.
pd.get_option('display.max_rows')
pd.set_option('display.max_columns', None)

# scaling the data 
from sklearn import preprocessing
mtcars_scale = preprocessing.scale(mtcars)

from scipy import stats
dir(stats.describe(mtcars_scale))
summ = stats.describe(mtcars_scale)
summ[1]
summ._fields
# summ = pd.DataFrame([summ], columns=summ._fields)

# Organized the scaled data into a DataFrame
mtcars_scale = pd.DataFrame(mtcars_scale)
mtcars_scale.columns = mtcars.columns
mtcars_scale.index = mtcars.index

import seaborn as sns; sns.set()
sns.heatmap(mtcars_scale)


### Creating a Heat Map for time series data
presidents = pd.read_csv('./data/presidents.csv', index_col=0)
sns.heatmap(presidents)


### Another way to create a Correlation Heat Map {corrplot}
import numpy as np
import matplotlib.pyplot as plt
# Compute the correlation matrix
corr = mtcars.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(h_neg=220, h_pos=10, as_cmap=True)
# h_neg, h_pos: Anchor hues for negative and positive extents of the map.
# as_cmap: If true, return a matplotlib colormap object rather than a list of colors.

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

### Creating a Choropleth Map
#copy to jupyder notebook
import plotly.graph_objects as go

# Load data frame and tidy it.
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2011_us_ag_exports.csv') # 美國各州農產品輸出統計表

fig = go.Figure(data=go.Choropleth(
    locations=df['code'], # Spatial coordinates
    z = df['total exports'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'Reds',
    colorbar_title = "Millions USD",
))

fig.update_layout(
    title_text = '2011 US Agriculture Exports by State',
    geo_scope='usa', # limite map scope to USA
)

fig.show()

