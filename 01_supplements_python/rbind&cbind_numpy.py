'''
Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the Dept. of ME and AI&DS (機械工程系與人工智慧暨資料科學研究中心主任), MCUT(明志科技大學); the IDS (資訊與決策科學研究所), NTUB (國立臺北商業大學); the CARS(中華R軟體學會創會理事長); and the DSBA(臺灣資料科學與商業應用協會創會理事長)
Notes: This code is provided without warranty.
'''

a = np.array([1,2,3])
b = np.array([4,5,6])
c = np.c_[a,b]

np.r_[a,b]

np.c_[c,a]

#### Reference
# numpy中np.c_和np.r_
