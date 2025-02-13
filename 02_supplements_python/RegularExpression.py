'''
Collated by Ching-Shih Tsou 鄒慶士 博士 (Ph.D.) Distinguished Prof. at the Department of Mechanical Engineering/Director at the Center of Artificial Intelligence & Data Science (機械工程系特聘教授兼人工智慧暨資料科學研究中心主任), MCUT (明志科技大學); Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), NTUB (國立臺北商業大學)
Notes: This code is provided without warranty.
'''

example = "1. A small sentence. - 2. Another tiny sentence."

import re

#### 1. Exact character matching
re.findall("small", example) # ['small']

re.findall("banana", example) # []

# Difference between re.search and re.findall
re.findall("sentence", example) # ['sentence', 'sentence']
re.search("sentence", example) # <re.Match object; span=(11, 19), match='sentence'>
example[11:19] # 'sentence'

example2 = ["text", "manipulation", "basics"]
[re.findall("a", string) for string in example2] # [[], ['a', 'a'], ['a']]

# Case sensitive
re.findall("SMALL", example) # []

re.findall("en", example) # ['en', 'en', 'en', 'en']

# Mixtures of alphabetic characters and blank spaces
re.findall("mall sent", example) # ['mall sent']

re.findall("2", example) # ['2']

re.findall("^2", example) # []

re.findall("sentence$", example) # []

re.findall("tiny|sentence", example) # ['sentence', 'tiny', 'sentence']

#### 2. Generalizing regular expressions
# . matches any character
re.findall("sm.ll", example) # ['small']

# [] character classes enclosed in brackets
re.findall("sm[abc]ll", example)  #['small']

# - concatenates range of characters
re.findall("sm[a-p]ll", example)  #['small']

re.findall("[uvw. ]", example) # ['.', ' ', ' ', ' ', '.', ' ', ' ', '.', ' ', ' ', ' ', '.']

#### 3. Predefined character classes
# [:digits:]
# [:lower:]
# [:upper:]
# [:alpha:]
# [:alnum:]
# [:punct:]: [!"\#$%&'()*+,\-./:;<=>?@\[\\\]^_‘{|}~]
# [:graph:]
# [:blank:]
# [:space:]
# [:print:]: [\x20-\x7E]{10}
# Since POSIX is not supported by Python re module, you have to emulate it with the help of character class.
re.findall("[\x20-\x7E]{10}", example)

#### 4. Quantifiers
re.findall("A.+sentence", example) # ['A small sentence. - 2. Another tiny sentence']
# atches any character (except for line terminators)
# + matches the previous token between one and unlimited times, as many times as possible, giving back as needed (greedy)
# sentence matches the characters sentence literally (case sensitive)

re.findall("A.+?sentence", example) # ['A small sentence', 'Another tiny sentence']
# . matches any character (except for line terminators)
# +? matches the previous token between one and unlimited times, as few times as possible, expanding as needed (lazy)
# sentence matches the characters sentence literally (case sensitive)

re.findall("(.en){1,5}", example) # ['ten', 'ten']
# 1st Capturing Group (.en){1,5}
# {1,5} matches the previous token between 1 and 5 times, as many times as possible, giving back as needed (greedy)
# A repeated capturing group will only capture the last iteration. Put a capturing group around the repeated group to capture all iterations or use a non-capturing group instead if you're not interested in the data
# . matches any character (except for line terminators)
# en matches the characters en literally (case sensitive)

# Match Information
# Match 1 11-17	senten
# Group 1 14-17	ten
# Match 2 39-45	senten
# Group 2 42-45	ten

re.findall(".en{1,5}", example) # ['sen', 'ten', 'sen', 'ten']
# . matches any character (except for line terminators)
# e matches the character e with index 10110 (6516 or 1458) literally (case sensitive)
# n matches the character n with index 11010 (6E16 or 1568) literally (case sensitive)
# {1,5} matches the previous token between 1 and 5 times, as many times as possible, giving back as needed (greedy)
# Match 1 11-14	sen
# Match 2 14-17	ten
# Match 3 39-42	sen
# Match 4 42-45	ten

#### 5. Metacharacters
# \w Word characters: [[:alnum:]]
# \W No word characters: [ˆ[:alnum:]]
# \s Space characters: [[:blank:]]
# \S No space characters: [ˆ[:blank:]]
# \d Digits: [[:digit:]]
# \D No digits: [ˆ[:digit:]]
# \b Word edge
# \B No word edge
# \< Word beginning
# \> Word end

re.findall("\.", example) # ['.', '.', '.', '.'] 標點符號嗎？No, matches the character . with index 4610 (2E16 or 568) literally (case sensitive).

re.findall("\w+", example) # ['1', 'A', 'small', 'sentence', '2', 'Another', 'tiny', 'sentence'] \w matches any word character (equivalent to [a-zA-Z0-9_])
# + matches the previous token between one and unlimited times, as many times as possible, giving back as needed (greedy)

pattern = "x+y*z"
re.findall(pattern, 'find xyz, xz, skip xyy, ww') # ['xyz', 'xz'],  matches the character x with index 12010 (7816 or 1708) literally (case sensitive)
# + matches the previous token between one and unlimited times, as many times as possible, giving back as needed (greedy) 一定要有x
# y matches the character y with index 12110 (7916 or 1718) literally (case sensitive)
# * matches the previous token between zero and unlimited times, as many times as possible, giving back as needed (greedy) y可有可無
# z matches the character z with index 12210 (7A16 or 1728) literally (case sensitive)

# Email 的 regular expression
with open('sample_emails2.txt','r') as d:
    sample_corpus = d.read()

sample_corpus[:500]
re.findall(r"\w*@.*\.com", sample_corpus)

# [A-Za-z0-9._]+@[A-Za-z.]+(com|edu)\.tw
# Please use (?:...) - a non-capturing version of regular parentheses.
re.findall(r"[A-Za-z0-9._]+@[A-Za-z.]+(?:com|edu)*", sample_corpus)

#### References:
# https://regex101.com
# Munzert, Simon, Rubba, Christian, Meißner, Peter, and Nyhuis, Dominic (2015), Automated Data Collection with R: A Practical Guide to Web Scraping and Text Mining, John Wiley & Sons.
# Python Regex: re.search() VS re.findall() (https://www.geeksforgeeks.org/python-regex-re-search-vs-re-findall/)
# Python RegEx: re.match(), re.search(), re.findall() with Example (https://www.guru99.com/python-regular-expressions-complete-tutorial.html)
# POSIX Bracket Expressions (https://www.regular-expressions.info/posixbrackets.html)
# Python: POSIX character class in regex? (https://stackoverflow.com/questions/31915346/python-posix-character-class-in-regex)
# re.findall not returning full match? (https://stackoverflow.com/questions/18425386/re-findall-not-returning-full-match)
# Regular Expression HOWTO (https://docs.python.org/3.3/howto/regex.html#Grouping)

