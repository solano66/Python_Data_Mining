'''
Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the IDS and CADS(資訊與決策科學研究所暨資料科學應用研究中心), NTUB(國立臺北商業大學); the CARS(中華R軟體學會); and the DSBA(臺灣資料科學與商業應用協會)
Notes: This code is provided without warranty.
'''

import seaborn as sns

### Qualitative palette 類型調色板
current_palette = sns.color_palette()
sns.palplot(current_palette)
# matplotlib六色調色板
themes = ['deep', 'muted', 'pastel', 'bright', 'dark', 'colorblind']
for theme in themes:
    current_palette = sns.color_palette(theme)
    sns.palplot(current_palette)

# hls (hue, saturation, lightness) color space
sns.palplot(sns.color_palette("hls", 8))
# 調亮度和飽和度
sns.palplot(sns.hls_palette(8, l=.3, s=.8))

### Sequential palette 順序調色板
sns.palplot(sns.color_palette("Blues"))

# 逆順序
sns.palplot(sns.color_palette("Blues_r"))

# 暗處理
sns.palplot(sns.color_palette("Blues_d"))

### Diverging palette 發散調色板

sns.palplot(sns.color_palette("BrBG", 7))

sns.palplot(sns.color_palette("RdBu_r", 7))

sns.palplot(sns.color_palette("coolwarm", 7))

# References:
# 顏色風格設置
# https://codertw.com/程式語言/664463/
