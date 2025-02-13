#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 05:28:20 2023

@author: Vince
"""

# For a function that returns a higher dimensional array, those dimensions are inserted in place of the axis dimension.
b = np.array([[1,2,3], [4,5,6], [7,8,9]])
b
np.apply_along_axis(np.diag, -1, b)
np.apply_along_axis(np.diag, 1, b)
np.apply_along_axis(np.diag, 0, b)