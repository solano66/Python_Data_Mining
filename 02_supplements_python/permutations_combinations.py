#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 20:54:10 2024

@author: vince
"""

from itertools import permutations, combinations  
permu = permutations(range(1,18),2)  
res1 = list(permu)  

comb = combinations(range(1,18),2)  
res2 = list(comb) 
