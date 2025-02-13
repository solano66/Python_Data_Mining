'''
Collated by Ching-Shih Tsou 鄒慶士 博士 (Ph.D.) Distinguished Prof. at the Department of Mechanical Engineering/Director at the Center of Artificial Intelligence & Data Science (機械工程系特聘教授兼人工智慧暨資料科學研究中心主任), MCUT (明志科技大學); Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), NTUB (國立臺北商業大學)
Notes: This code is provided without warranty.
'''

#### 1.4 Python語言原生資料物件操弄
#### (補充) 字串基礎 Strings Fundamentals
#### 簡單字串 Simple string
a = 'this is a string'
type(a)

a = 'this is the first half '
b = 'and this is the second half'
a + b

simple_string = 'hello' + " I'm a simple string"
simple_string

#### 取值與不可更動 Subsetting and immutable
print(a[10])
a[10] = 'f' # TypeError: 'str' object does not support item assignment

#### 型別轉換 Type conversion
a = 5.6
type(a)
s = str(a)
s
type(s)

#### 多列字串 Multi-line string, note the \n (newline) escape character automatically created
multi_line_string = """Hello I'm
a multi-line
string!"""
multi_line_string
# Attention to the difference to above
print(multi_line_string)

#### 路徑字串 Normal string with escape sequences leading to a wrong file path!
escaped_string = "C:\the_folder\new_dir\file.txt"
print(escaped_string) # will cause errors if we try to open a file, because \t, \n, and \f here

# raw string keeping the backslashes in its normal form
raw_string = r'C:\the_folder\new_dir\file.txt'
print(raw_string)


#### 萬國碼與字元碼 Unicode and bytecode string literals
# https://unicode-table.com/cn/2639/
smiley = u"\u263A"

print(smiley)

type(smiley) # str

(u"\u263A").decode()
# AttributeError: 'str' object has no attribute 'decode'

ord(smiley) # = 9786 decimal value of code point = 263A in Hex. (Return the Unicode code point for a one-character string.)

len(smiley) # 1

smiley.encode('utf8') # prints '\xe2\x98\xba' the bytes - it is <str>

type(smiley.encode('utf8')) # bytes

print (b'\xe2\x98\xba')

(b'\xe2\x98\xba').decode() # A smiling face

len(smiley.encode('utf8')) # its length = 3, because it is encoded as 3 bytes

print(u"\u263A".encode('ascii')) # 'ascii' codec can't encode character '\u263a' in position 0: ordinal not in range(128)
 
string_with_unicode = u'H\u00e8llo!'
print(string_with_unicode)
ord(string_with_unicode)
# TypeError: ord() expected a character, but string of length 5 found

string_with_unicode = u'H\xe8llo'
print(string_with_unicode)
ord(string_with_unicode)
# TypeError: ord() expected a character, but string of length 5 found

u'H\u00e8llo!'.encode('utf8') # b'H\xc3\xa8llo!'
u'H\u00e8llo!'.encode() # b'H\xc3\xa8llo!'

#### 字串格式化 String formatting
template = '%.2f %s are worth $%d'
template % (31.5560, 'Taiwan Dollars', 1)

template = '{:.2f} {:s} are worth ${:d}'
template.format(31.5560, 'Taiwan Dollars', 1)

#### Python 3 逃脫字串 Python 3 Escape Sequences
# http://python-ds.com/python-3-escape-sequences)
# Unless an `r' or `R' prefix is present, escape sequences in strings are interpreted according to rules similar to those used by Standard C.

# \n means line break
'Hel\nlo'
print('Hel\nlo')

# \newline Backslash and newline ignored
print("line1 \
line2 \
line3")

# \\ Backslash (\)
print("\\")

# \' Single quote (')
print('\'')

# \" Double quote (")
print("\"")

# \b ASCII Backspace (BS)
print("Hello \b World!")

# \f ASCII Formfeed (FF)
print("Hello \f World!")

# \n ASCII Linefeed (LF)
print("Hello \n World!")

# \r ASCII Carriage Return (CR)
print("Hello \r World!")

# \t ASCII Horizontal Tab (TAB)
print("Hello \t World!")

# \v ASCII Vertical Tab (VT)
print("Hello \v World!")

# Character with octal value \ooo ooo
print("\110\145\154\154\157\40\127\157\162\154\144\41")

# Character with hex value \xhh hh
print("\x48\x65\x6c\x6c\x6f\x20\x57\x6f\x72\x6c\x64\x21")

s = '12\\34'
print(s)
s

# Using "r" to prevent resolution
r'Hel\nlo'
print(r'Hel\nlo')

s = r'this\has\no\special\characters'
s
print(s)

#### (補充)Operations and Operators
1 + 1 == 2
1 + 1 is 2
8 + 7 == 87
8 + 9 is not 1
"放生" == "棄養"
False or True
False and True
False | True
False & True
bool('Hello world!')
bool('')
bool(0)
bool(1)

USD = 31.3987
JPY = 0.2738
USD * JPY + JPY/USD + USD **2 + JPY ** (1/2)

JPY ** (1/2)

