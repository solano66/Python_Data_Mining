## Collated by Ching-Shih Tsou 鄒慶士 博士 (Ph.D.) Distinguished Prof. at the Department of Mechanical Engineering/Director at the Center of Artificial Intelligence & Data Science (機械工程系特聘教授兼人工智慧暨資料科學研究中心主任), MCUT (明志科技大學); Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), NTUB (國立臺北商業大學)
## Notes: This code is provided without warranty.

## 4.3 集群分析 ####
## 4.3.2 階層式集群 ------------------------------------------------------------------------
# R語言美國汽車雜誌道路測試資料
str(mtcars)

## ------------------------------------------------------------------------
# 階層式集群要先產生觀測值間的距離值矩陣(*輸出為向量物件*)
d <- dist(mtcars)

## ----fig.align="center", fig.cap = "\\label{fig:dendrogram}階層式集群的樹狀圖"----
# 根據距離進行聚合法階層式集群
# 群間距離計算方法預設為最遠距離法(complete)
hc <- hclust(d) # Why 496 elements ? (32*32 - 32)/2
# class(hc)
# 繪製樹狀圖
# plot(hc)
plot(hc, hang = -1)
# ?plot.hclust

## ------------------------------------------------------------------------
# 類別名稱與產製函數同名
class(d)

## ----eval=FALSE----------------------------------------------------------
## # 距離物件d的結構，為何是496個元素？
## str(d)

## ----warning=FALSE, message=FALSE----------------------------------------
# 載入Clustering for Business Analytics套件
library(cba)
# 注意列名與行名
subset(d, 1:5)
rownames(mtcars)[1:5]

## ----eval=FALSE----------------------------------------------------------
## # args()函數返回距離計算函數dist()的引數及其預設值
## args(dist)

## ------------------------------------------------------------------------
# 階層式集群模型物件hc的內容
names(hc)
# 第一橫列-1與-2表示第一次聚合(merge)編號1和2的樣本
hc$merge
rownames(mtcars)[1:2]
# 第四次聚合是編號14的樣本與前面第二群中編號12與13的樣本
rownames(mtcars)[14]
rownames(mtcars)[12:13]
# 每次聚合對象間的距離值，總共聚合31次
hc$height

