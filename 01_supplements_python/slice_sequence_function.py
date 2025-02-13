#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 12:30:49 2024

@author: vince
"""

# from string import ascii_lowercase
# from random import choice

# letters = [choice(ascii_lowercase) for _ in range(5)]

y = [chr(x) for x in range(97, 105)]
x = slice(2)
print(y[x])

x = slice(3, 5)
print(y[x])

x = slice(0, 8, 3)
print(y[x]) # 0, 0+3, 0+3+3, so a, d, and g have been extracted
