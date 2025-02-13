'''
Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the IDS Institute and ICD Lab. of NTUB (臺北商業大學資訊與決策科學研究所暨智能控制與決策研究室教授); at the ME Dept. and CAIDS of MCUT (曾任明志科技大學機械工程系特聘教授兼人工智慧暨資料科學研究中心主任); at the IM Dept. of SHU (曾任世新大學資訊管理學系副教授); at the BA Dept. of CHU (曾任中華大學企業管理學系副教授); the CSQ (中華民國品質學會AI暨大數據品質應用委員會主任委員); the DSBA (2013年創立臺灣資料科學與商業應用協會); and the CARS (2012年創立中華R軟體學會)
Notes: This code is provided without warranty.
'''

import numpy as np
from sklearn.cluster import KMeans

def RandIndex(cluser, gt):
    TP=0
    TN=0
    FP=0
    FN=0
    
    for i in range(0, len(cluser)-1):
        for j in range(i+1, len(cluser)):
            if (cluser[i]==cluser[j]):
                # Positive pair
                if (gt[i]==gt[j]):
                    # True
                    TP=TP+1
                else:
                    # False
                    FP=FP+1
            else:
                # Negative pair
                if (gt[i]==gt[j]):
                    # False
                    FN=FN+1
                else:
                    # True
                    TN=TN+1
                    
    Rand = (TP+TN)/(TP+TN+FP+FN)
    return Rand

if __name__=="__main__":
    
    rawData = [[7, 15, 20, 20],
               [5, 3, 2, 1],
               [4, 3, 0, 4],
               [11, 12, 7, 8],               
               [15, 20, 7, 20]]
    
    data = np.array(rawData)
    
    k = 3
    
    km = KMeans(n_clusters=k).fit(data)
    clusteringID = km.labels_
    # clusteringID = [0, 0, 1, 2, 1]
    groundTruth = [2, 2, 3, 1, 3]
    
    print("Clustering result:\n", clusteringID)
    print("Ground truth:\n", groundTruth)
    
    rand = RandIndex(clusteringID, groundTruth)
    print("Rand index", rand)
                    