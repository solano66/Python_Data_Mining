########################################################## Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the IDS and CADS(資訊與決策科學研究所暨資料科學應用研究中心), NTUB(國立臺北商業大學); the CARS(中華R軟體學會); and the DSBA(臺灣資料科學與商業應用協會)
#######################################################
### Notes: This code is provided without warranty.

import pandas as pd
df = pd.DataFrame({'Gender':['f', 'f', 'm', 'f', 'f', 'm', 'm', 'f', 'm', 'f', 'm'], 'TV': [3.4, 3.5, 2.6, 4.7, 4.2, 4.2, 5.1, 3.9, 3.7, 2.1, 4.3]})

df

df['Gender'][3:7] # 3, 4, 5, 6
df[['TV', 'Gender']][3:7] # 3, 4, 5, 6
df[3:7][['TV', 'Gender']] # 3, 4, 5, 6
help(df.iloc)
df.iloc[3:7, 0:2] # 連續位置取值 3, 4, 5, 6 & 0, 1

df.iloc[[3,5], 0:2] # 間斷位置取值 3, 5 & 0, 1

df.loc[3:7, 'Gender'] # 3, 4, 5, 6, 7*
df.loc[3:7, ['TV']] # 3, 4, 5, 6, 7* with variable name 'TV'

dir(df)
grouped = df.groupby('Gender')
grouped
list(grouped)
dir(grouped)

grouped.describe()
help(grouped.get_group)
# grouped.get_group("M")
grouped.get_group("m")

grouped.boxplot()

