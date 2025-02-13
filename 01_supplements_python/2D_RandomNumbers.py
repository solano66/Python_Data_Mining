'''
Collated by Ching-Shih Tsou 鄒慶士 博士 (Ph.D.) Distinguished Prof. at the Department of Mechanical Engineering/Director at the Center of Artificial Intelligence & Data Science (機械工程系特聘教授兼人工智慧暨資料科學研究中心主任), MCUT (明志科技大學); Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), NTUB (國立臺北商業大學)
Notes: This code is provided without warranty.
'''

N = 5
import random
# A 5 x 5 nested list
[[random.random() for i in range(N)] for j in range(N)]

import numpy as np
# A numpy ndarray with shape (5, 5)
np.random.random((N,N))

np.random.randint(5, size=(N,N)) # array([4, 3])

### Regerence: Simple way of creating a 2D array with random numbers (Python) (https://stackoverflow.com/questions/24108417/simple-way-of-creating-a-2d-array-with-random-numbers-python)