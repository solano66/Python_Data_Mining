import pandas as pd
from sklearn.cluster import DBSCAN

teens = pd.read_csv("./snsdata.csv")
# 'gradyear'不需要用於計算, 所以轉換成'str'的型態
teens['gradyear'] = teens['gradyear'].astype('str')


# 1. 把數值資料標準化
from sklearn.preprocessing import scale
teens_z=scale(teens.iloc[:,4:])

# 2. 用DBSCAN分群(建立空模、配適)
mdl = DBSCAN(eps=5, min_samples=5)
mdl.fit(teens_z)
print(set(mdl.labels_)) #確認資料被分成幾個群, '-1'是噪音

# 3.從模型中抓出各群的索引
labels=mdl.labels_

# 4.使用群索引來選取標準化後的資料中每個群的資料，但只包括特徵值的部分
grouped_data = {}   
#↑用dict來儲存{群索引:群內的x筆資料的y個特徵值}, 
#也就是說, key=群索引, value=x筆*36個特徵值
for label in set(labels):
    if label == -1:
        continue  # 跳過噪音點
    grouped_data[label] = teens_z[labels == label]  
    #↑這行會自動把labels跟teens_z對應起來: 
    #labels的row, 會被對應到teens_z的row, 
    #如果labels.row=label, 則對應的teens_z.row會放進gouped_data裡面
print(grouped_data.keys())  
#↑可以看到dict裡面已經有5個key, 也就是5個群(0,1,2,3,4)
#每個key對應的value, 其shape為[x*36]，也就是有x筆資料*36個標準化的特徵值

# 5. 對於每個群，計算其所有資料的 36 個特徵值的平均值，即特徵中心點
cluster_centers = {}
for label, group in grouped_data.items():
    #label會抓dict的key, group會抓dict的value(shape=[x*36])
    cluster_centers[label] = pd.DataFrame(group).mean()
    
# 把cluster_centers從dict轉成df 
# (轉成df才能幫columns加上label, 後續繪圖的可讀性會比較好)
cens=pd.DataFrame.from_dict(cluster_centers, orient="index")
# 幫cens的columns加上label
colLabel=teens.iloc[:,4:].columns
cens.columns=colLabel



# 各群中心座標矩陣轉置後繪圖
ax = cens.T.plot() # seaborn, ggplot or pandas ?
# 低階繪圖設定x 軸刻度位置
ax.set_xticks(list(range(len(cens.T))))
# 低階繪圖設定x 軸刻度說明文字
ax.set_xticklabels(list(cens.columns), rotation=90)
fig = ax.get_figure()
fig.tight_layout()
fig
# fig.savefig('./_img/sns_lineplot.png')    

# 添加群編號於原資料表後
teens = pd.concat([teens, pd.Series(mdl.labels_).rename('cluster')], axis=1)

# 抓集群未使用的三個變量(剛才歸群時未用，但事後分析確有助於了解各群的異同，以及歸群結果的品質)
teens[['gender','age','friends','cluster']][0:5]

# 各群平均年齡(群組與摘要也！)
teens.groupby('cluster').aggregate({'age': "mean"}) # 同儕間年齡差異不大！

# 新增是否為女生欄位'female'
teens.gender.value_counts()
teens.gender.value_counts(dropna = False)


