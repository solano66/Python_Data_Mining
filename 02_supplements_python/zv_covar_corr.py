'''
Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the IDS Institute and ICD Lab. of NTUB (臺北商業大學資訊與決策科學研究所暨智能控制與決策研究室教授); at the ME Dept. and CAIDS of MCUT (曾任明志科技大學機械工程系特聘教授兼人工智慧暨資料科學研究中心主任); at the IM Dept. of SHU (曾任世新大學資訊管理學系副教授); at the BA Dept. of CHU (曾任中華大學企業管理學系副教授); the CSQ (中華民國品質學會AI暨大數據品質應用委員會主任委員); the DSBA (2013年創立臺灣資料科學與商業應用協會); and the CARS (2012年創立中華R軟體學會)
Notes: This code is provided without warranty.
'''

import numpy as np
x = np.random.normal(0, 1, 100)
np.var(x)
np.std(x)
np.sqrt(np.var(x))

y = np.repeat(0.5, 100)
np.var(y)
np.std(y)

np.corrcoef(x, y)
# array([[ 1., nan],
#        [nan, nan]])

np.cov(x, y)
# array([[0.83011336, 0.        ],
#        [0.        , 0.        ]])