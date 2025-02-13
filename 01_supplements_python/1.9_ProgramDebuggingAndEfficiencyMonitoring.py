'''
Collated by Ching-Shih Tsou 鄒慶士 博士 (Ph.D.) Distinguished Prof. at the Department of Mechanical Engineering/Director at the Center of Artificial Intelligence & Data Science (機械工程系特聘教授兼人工智慧暨資料科學研究中心主任), MCUT (明志科技大學); Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), NTUB (國立臺北商業大學)
Notes: This code is provided without warranty.
'''

### 1.9 程式除錯與效率監測
# 數學函數不能施加在字串上
import numpy as np
np.log(['a', 'b', 'c'])

np.log(['2', '3', '4']) # casting='same_kind'

# 有兩則警告訊息
np.log([-1, 0, 1, 2])

# 壓制警告訊息的顯示
import warnings
warnings.filterwarnings("ignore")
np.log([-1, 0, 1, 2])

# 運用程式碼來偵測錯誤
while True:
    try:
        x = int(input("Please enter a number: "))
        print("The number you input is {}.".format(x))
        break
    except ValueError:
        print("Oops!  That was no valid number.  Try again please ...")

# 運用traceback()獲取額外的錯誤訊息(改成https://www.geeksforgeeks.org/traceback-in-python/)
import traceback

try:
    1/0
except Exception as e:
    traceback.print_exc()

# NameError: name 'NaN' is not defined
0/NaN


# 計算程式執行時間
import random
import time
start_time = time.time()

# 輸入的向量中，奇數元素的個數(%表除法運算後取餘數)
a = []
def oddcount(x):
    if (x%2==1):
        a.append(x)
    return len(a)

for i in range(1,10000):
    x = random.randint(1,1000000)
    print(oddcount(x))

print("%s seconds" % (time.time() - start_time))


## 自訂函數cv(x)計算向量x各元素除以平均值後的標準差
import statistics
import numpy as np
def cv(x):
    print('In cv, x = ', x, '\n')
    print('mean(x)=', np.mean(x), '\n')
    print('sd(x/mean(x))=', np.std(x/np.mean(x)), '\n')

for i in range(1, 3):
    cv(i)

cv([1,2,3])

cv("0")

for i in ["a", "b", "c"]:
    cv(i)

# 前面運用traceback()獲取額外的錯誤訊息
try:
    cv("0")
except Exception as e:
    traceback.print_exc()

# 在函式中放ipdb.set_trace()進入偵錯模式
import ipdb # !conda install -c conda-forge ipdb --y
import statistics
import numpy as np

def cv(x):
    ipdb.set_trace()
    print('In cv, x = ', x, '\n')
    print('mean(x)=', np.mean(x), '\n')
    print(np.std(x/np.mean(x)))

for i in range(1, 3):
    cv(i)

for i in ["a", "b", "c"]:
    cv(i)

