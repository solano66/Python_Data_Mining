"""
Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the IDS, CICD, and CISD of NTUB (臺北商業大學資訊與決策科學研究所暨智能控制與決策研究室教授兼校務永續發展中心主任); at the ME Dept. and CAIDS of MCUT (2020年借調至明志科技大學機械工程系擔任特聘教授兼人工智慧暨資料科學研究中心主任兩年); the CSQ (2019年起任中華民國品質學會大數據品質應用委員會主任委員); the DSBA (2013年創立臺灣資料科學與商業應用協會); and the CARS (2012年創立中華R軟體學會)
Notes: This code is provided without warranty.
"""

import statsmodels.formula.api as smf


def forward_selected(data, response):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model


import pandas as pd

url = "http://data.princeton.edu/wws509/datasets/salary.dat"
data = pd.read_csv(url, sep='\\s+') # (52, 6)

data.dtypes

model = forward_selected(data, 'sl')

type(model)
dir(model)
[(nam, type(getattr(model, nam))) for nam in dir(model)] 

print(model.model.formula)
# sl ~ rk + yr + 1

print(model.rsquared_adj)

#### Another example of Solubility
solTrainX = pd.read_csv('solTrainX.csv',encoding='utf-8')
solTrainXtrans = pd.read_csv('solTrainXtrans.csv',encoding='utf-8') # 整數值變量帶小數點
solTrainY = pd.read_csv('solTrainY.csv',encoding='utf-8')

sol = pd.concat([solTrainXtrans, solTrainY], axis='columns')
sol.rename(columns={'x':'solubility'}, inplace=True)

# Long, long, long ... execution time
model = forward_selected(sol, 'solubility')

print(model.model.formula) # 139 variables selected
# solubility ~ MolWeight + SurfaceArea1 + NumNonHAtoms + FP142 + FP074 + FP206 + FP137 + FP172 + FP173 + FP002 + NumMultBonds + FP116 + FP049 + FP083 + FP085 + FP135 + FP164 + FP202 + FP188 + FP124 + FP004 + FP026 + FP059 + FP040 + FP127 + NumCarbon + FP039 + FP190 + FP037 + FP154 + FP111 + FP075 + FP129 + FP056 + FP204 + NumHydrogen + NumSulfer + FP084 + NumAtoms + FP078 + FP027 + FP022 + FP071 + FP061 + FP099 + NumBonds + NumNonHBonds + FP076 + FP044 + FP122 + FP079 + FP147 + FP176 + FP163 + FP064 + FP081 + FP093 + NumRotBonds + FP171 + NumChlorine + FP128 + FP109 + NumOxygen + NumNitrogen + FP201 + FP096 + FP072 + FP065 + FP119 + FP184 + FP107 + FP077 + FP126 + FP131 + FP054 + FP069 + FP098 + FP140 + FP103 + FP113 + FP169 + FP174 + FP167 + FP165 + NumDblBonds + FP066 + FP134 + FP019 + FP018 + FP055 + FP150 + NumAromaticBonds + FP005 + FP089 + FP068 + FP145 + FP157 + FP067 + FP088 + FP104 + FP051 + FP118 + FP052 + SurfaceArea2 + FP143 + FP130 + FP159 + FP032 + FP033 + FP017 + FP156 + FP045 + FP048 + FP094 + FP161 + FP182 + FP070 + FP073 + FP038 + NumRings + FP080 + FP185 + FP024 + FP101 + FP031 + FP023 + FP016 + FP063 + FP186 + FP133 + FP191 + FP153 + FP035 + FP020 + FP015 + FP115 + FP148 + FP177 + FP062 + 1

print(model.rsquared_adj) # 0.9314400624342031

model.save('solubility_forward_statsmodels.pickle')
import statsmodels.api as sm
new_model = sm.load('solubility_forward_statsmodels.pickle')
new_model.summary()
type(new_model.summary())

dir(new_model.summary())
new_model.summary().tables
# [<class 'statsmodels.iolib.table.SimpleTable'>,
# <class 'statsmodels.iolib.table.SimpleTable'>,
# <class 'statsmodels.iolib.table.SimpleTable'>]

print(new_model.summary().tables[1])

with open('solubility_forward_statsmodels_summary.csv', 'w') as fh:
    fh.write(new_model.summary().as_csv())

new_model.summary().as_csv() # The output of as_csv() is not machine‑readable.

#### Reference:
# https://planspace.org/20150423-forward_selection_with_statsmodels/
# https://stackoverflow.com/questions/16420407/python-statsmodels-ols-how-to-save-learned-model-to-file
# https://tw.coderbridge.com/questions/b1f0c3fffaae48919a90a149f87fea75
