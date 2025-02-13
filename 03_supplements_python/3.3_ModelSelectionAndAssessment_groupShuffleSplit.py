#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import GroupShuffleSplit
X = np.ones(shape=(8, 2))
y = np.ones(shape=(8, 1))
groups = np.array([1, 1, 2, 2, 2, 3, 3, 3])
print(groups.shape)
gss = GroupShuffleSplit(n_splits=2, train_size=.7, random_state=42)
gss.get_n_splits()
print(gss)
GroupShuffleSplit(n_splits=2, random_state=42, test_size=None, train_size=0.7)

for i, (train_index, test_index) in enumerate(gss.split(X, y, groups)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index},   group={groups[train_index]}")
    print(f"  Test:  index={test_index}, group={groups[test_index]}")