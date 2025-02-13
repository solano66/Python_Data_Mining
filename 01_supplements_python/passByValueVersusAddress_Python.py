'''
Collated by Ching-Shih Tsou 鄒慶士 博士 (Ph.D.) Distinguished Prof. at the Department of Mechanical Engineering/Director at the Center of Artificial Intelligence & Data Science (機械工程系特聘教授兼人工智慧暨資料科學研究中心主任), MCUT (明志科技大學); Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), NTUB (國立臺北商業大學)
Notes: This code is provided without warranty.
'''

#### 傳值、傳址的概念和區別：
# 傳值：被調函數局域變數改變不會影響主調函數局域變數
# 傳址：被調函數局域變數改變會影響主調函數局域變數

# 傳值就是傳入一個引數的值，傳址就是傳入一個引數的地址，也就是記憶體的地址（相當於指標）。他們的區別是如果函數裡面對傳入的引數重新賦值，函數外的全域性變數是否相應改變，用傳值傳入的引數是不會改變的，用傳址傳入就會改變。

a = 1

def f(b):
    b = 2
    return b

f(a)

print(a)

# 在這段程式碼裡面，首先宣告a的值為1，把a作為引數傳入到函式f裡面，函式f裡面對b重新賦值為2，如果是傳值的形式傳入a的話，a的值是不會變的，依然為1，如果以傳址的形式（但是在python中這個不是程式設計師能決定的）傳入a，a就會變成2。這個就是傳值和傳址的區別。

#### Python 引數傳遞的方式
# Python是不允許程式設計師選擇採用傳值還是傳址的。Python引數傳遞採用的肯定是“傳物件引用”的方式。實際上，這種方式相當於傳值和傳址的一種綜合。

# 如果函式收到的是一個可變物件（比如字典或者列表）的引用，就能修改物件的原始值——相當於傳址。如果函式收到的是一個不可變物件（比如數字、字元或者元組）的引用（其實也是物件地址！！！），就不能直接修改原始物件——相當於傳值。

# 所以python的傳值和傳址是比如根據傳入引數的型別來選擇的

# 傳值的引數型別：數字，字串，元組（immutable）

# 傳址的引數型別：列表，字典（mutable）

#### Python的傳值
a = 1

def f(b):
    b = b + 1
    return b

f(a)

print(a)
# 這段程式碼裡面，因為a是數字型別，所以是傳值的方式，a的值並不會變，輸出為1。

#### Python的傳址
a = [1]

def f(b):
    b[0] = b[0] + 1
    return b

f(a)

print(a)
# 這段程式碼裡面，因為a的型別是列表，所以是傳址的形式，a[0]的值會改變，輸出為[2]。

#### Python的傳址
def try_to_change_list_contents(the_list):
    print('got', the_list)
    the_list.append('four')
    print('changed to', the_list)
 
outer_list = ['one', 'two', 'three']
 
print('before, outer_list =', outer_list)
try_to_change_list_contents(outer_list)
print('after, outer_list =', outer_list)

#### Python的傳值
def try_to_change_string_reference(the_string):
    print('got', the_string)
    the_string = 'In a kingdom by the sea'
    print('set to', the_string)
 
outer_string = 'It was many and many a year ago'
 
print('before, outer_string =', outer_string)
try_to_change_string_reference(outer_string)
print('after, outer_string =', outer_string)

#### Copy and Deepcopy
# copy使用場景：列表或字典，且內部元素為數字，字串或元組
# deepcopy使用場景：列表或字典，且內部元素包含列表或字典

#### References:
# https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/531439/
# https://e8859487.pixnet.net/blog/post/402632162-python-%e5%82%b3%e5%80%bc%28pass-by-value%29-vs-%e5%82%b3%e5%9d%80%28pass-by-address%29-vs-
# https://ifun01.com/8DMO5FK.html

