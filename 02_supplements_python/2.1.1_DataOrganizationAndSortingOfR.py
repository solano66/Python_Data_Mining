'''
Collated by Ching-Shih Tsou 鄒慶士 博士 (Ph.D.) Distinguished Prof. at the Department of Mechanical Engineering/Director at the Center of Artificial Intelligence & Data Science (機械工程系特聘教授兼人工智慧暨資料科學研究中心主任), MCUT (明志科技大學); Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), NTUB (國立臺北商業大學)
Notes: This code is provided without warranty.
'''

#### 2.1.1 R語言資料組織與排序
import pandas as pd
# 創建姓名與成績向量
Student = ["John Davis", "Angela Williams", "Bullwinkle Moose", "David Jones", "Janice Markhammer", "Cheryl Cushing", "Reuven Ytzrhak", "Greg Knox", "Joel England", "Mary Rayburn"]  
Math = [502, 600, 412, 358, 495, 512, 410, 625, 573, 522]
Science = [95, 99, 80, 82, 75, 85, 80, 95, 89, 86]
English = [25, 22, 18, 15, 20, 28, 15, 30, 27, 18]

# 從原生串列合併為原生字典
dic = {'Student': Student,
        'Math': Math,
        'Science': Science,
        'English': English
        }
# 再從原生字典組織成衍生的二維資料框物件roster
roster = pd.DataFrame(dic)
print(roster)

# sklearn.preprocessing的尺度調整(標準化)函數
# Step 1 import necessary packages
from sklearn.preprocessing import StandardScaler
# Step 2 define an empty model
scale = StandardScaler()
# Steps 3 input training data to fit the model & 4 transform and apply
z = scale.fit_transform(roster.iloc[:,1:4]) # DDP常常輸入輸出不同調！
z = pd.DataFrame(z)
# 標準化後各科平均數非常接近0，各科標準差接近1
z.describe().T

# 計算(十位同學)三科平均成績，新增到roster的最末行
score = z.mean(axis=1)
roster['score'] = score
# pandas Series的統計位置量數計算函數quantile()
y = roster['score'].quantile([.8,.6,.4,.2])

#     y.iloc[0]被循環利用(broadcasting, recycling)(短物件中的元素循環被利用)
score >= y.iloc[0]

# 不用宣告即可直接新增grade欄位
# 百分比等第排名值從高到低依序填入grade 欄位(資料操弄時多採用邏輯值索引，避免使用if…then…條件式語法)
roster['grade'] = ''

roster.loc[:,'grade'][score >= y.iloc[0]] = "A"

roster.loc[:,'grade'][(score < y.iloc[0]) & (score >= y.iloc[1])] = "B"

roster.loc[:,'grade'][(score < y.iloc[1]) & (score >= y.iloc[2])] = "C"

roster.loc[:,'grade'][(score < y.iloc[2]) & (score >= y.iloc[3])] = "D"

roster.loc[:,'grade'][score < y.iloc[3]] = "F"

# 以空白斷開名與姓
name = roster.loc[:,'Student'].str.split(' ')
# 鏈式索引
name[0][0]
name[0][1]

lastname = []
for i in range(len(name)):
    lastname.append(name[i][1]) # 想想name[0][1]

firstname = []
for i in range(len(name)):
    firstname.append(name[i][0]) # 想想name[0][0]

# 移除原Student欄位，添加firstname與lastname兩欄位
roster = roster.drop(labels=['Student'], axis='columns')
roster['firstname'] = firstname
roster['lastname'] = lastname

# 表格排序，注意列首的觀測值編號(index)
roster = roster.sort_values(by=['lastname','firstname'])
roster = roster[['lastname', 'firstname', 'Math', 'Science', 'English', 'score', 'grade']]



