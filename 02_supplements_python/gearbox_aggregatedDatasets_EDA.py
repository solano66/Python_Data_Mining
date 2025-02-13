
# About Dataset
# Context
# There are few dataset on mechanical engineering, in particular devoted to apply Machine Learning in industrial environment. This dataset were not yet present in Kaggle. So it's good for the community to have it at hand.

# Content
# Gearbox Fault Diagnosis Data set include the vibration dataset recorded by using SpectraQuest’s Gearbox Fault Diagnostics Simulator. 變速箱故障診斷模擬器SpectraQuest
# Dataset has been recorded using 4 vibration sensors placed in four different direction, and under variation of load from '0' to '90' percent. 不同方向的四個振動感測器，負載從0%遞增到90%，間距10% Two different scenario are included:
# 1) Healthy condition and
# 2) Broken Tooth Condition
# There are 20 files in total, 10 for healthy gearbox and 10 from broken one. Each file corresponds to a given load from 0% to 90% in steps of 10%.

# Acknowledgements
# Dataset taken from https://openei.org/datasets/dataset/gearbox-fault-diagnosis-data

#### Read New Aggregated Dataset
import pandas as pd
gb = pd.read_csv('./broken_health_aggregated.csv', index_col=0)
gb.dtypes
gb.load.value_counts().sort_index()
gb.failure.value_counts().sort_index()

pd.crosstab(gb.failure, gb.load)

#### Reference:
# https://www.kaggle.com/code/melikedemirdag/gearbox-fault-aggregated-datasets