#!/usr/bin/env python3
# -*- coding: utf-8 -*-
### Collated by Ching-Shih Tsou 鄒慶士 博士 (Ph.D.) Distinguished Prof. at the Department of Mechanical Engineering/Director at the Center of Artificial Intelligence & Data Science (機械工程系特聘教授兼人工智慧暨資料科學研究中心主任), MCUT (明志科技大學); Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), NTUB (國立臺北商業大學)
### Notes: This code is provided without warranty.

import pandas as pd

# A nested list 巢狀/嵌套式串列 for transactions
items = [["Bread", "Milk"], ["Bread", "Diaper", "Beer", "Eggs"], ["Milk", "Diaper", "Beer", "Coke"], ["Bread", "Milk", "Diaper", "Beer"], ["Bread", "Milk", "Diaper", "Coke"]]

from mlxtend.preprocessing import TransactionEncoder # !conda install -c conda-forge mlxtend --y
te = TransactionEncoder()
txn_binary = te.fit(items).transform(items)
print(txn_binary.shape)
dir(te) # See 'columns_mapping_' and 'columns'
te.columns_
te.columns_mapping_
df = pd.DataFrame(txn_binary, columns=te.columns_)

from mlxtend.frequent_patterns import apriori
freq_itemsets = apriori(df, min_support=0.4, use_colnames=True) # 5 * 0.4 = 2 transactions at least (inclusive)


from mlxtend.frequent_patterns import association_rules
itemsrules = association_rules(freq_itemsets, metric="confidence", min_threshold=0.5)
