#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

df = pd.DataFrame(['a','b','c','d','a','b'])

df

df.duplicated(keep=False)
df[df.duplicated(keep=False)]
df[~df.duplicated(keep=False)]

df.duplicated(keep="first")
df[~df.duplicated(keep="first")]

df.duplicated(keep="last")
df[~df.duplicated(keep="last")]

#### Reference:
# https://stackoverflow.com/questions/14657241/how-do-i-get-a-list-of-all-the-duplicate-items-using-pandas-in-python
