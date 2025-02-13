## Collated by Ching-Shih Tsou 鄒慶士 博士 (Ph.D.) Distinguished Prof. at the Department of Mechanical Engineering/Director at the Center of Artificial Intelligence & Data Science (機械工程系特聘教授兼人工智慧暨資料科學研究中心主任), MCUT (明志科技大學); Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), NTUB (國立臺北商業大學)
## Notes: This code is provided without warranty.

## 2.1 資料管理 ####
## 2.1.1 R語言資料組織與排序 ------------------------------------------------------------------------
# 創建姓名與成績向量，注意指令有無包在小括弧裡的差異
(Student <- c("John Davis", "Angela Williams",
"Bullwinkle Moose", "David Jones", "Janice Markhammer",
"Cheryl Cushing", "Reuven Ytzrhak", "Greg Knox",
"Joel England", "Mary Rayburn"))
(Math <- c(502, 600, 412, 358, 495, 512, 410, 625, 573, 522))
Science <- c(95, 99, 80, 82, 75, 85, 80, 95, 89, 86)
English <- c(25, 22, 18, 15, 20, 28, 15, 30, 27, 18)
# 組織二維資料框roster
(roster <- data.frame(Student, Math, Science, English,
stringsAsFactors = FALSE))

## ------------------------------------------------------------------------
# 尺度調整函數，留意標準化計算所需之各科平均數與標準差
(z <- scale(roster[, 2:4])) # vectorization !
# z類別為數值型矩陣
mode(z)
class(z)
# 標準化後各科平均數非常接近0 (by Implicit Looping)
apply(z, MARGIN = 2, FUN = mean) # MARGIN is the axis argument in Python
# 標準化後各科標準差為1 (by Implicit Looping)
apply(z, 2, sd)

## ------------------------------------------------------------------------
# 計算三科平均成績，並以cbind()併入roster最末行
score <- apply(z, 1, mean)
(roster <- cbind(roster, score)) # 類似numpy.concatenate()與pandas.concat()

## ------------------------------------------------------------------------
# 統計位置量數計算函數quantile()
(y <- quantile(score, probs = c(.8,.6,.4,.2)))
# y[1]被循環利用(recycled in R or broadcasting in Python)(短物件中的元素循環被利用)
score >= y[1]
# 不用宣告即可直接新增grade欄位
# 百分比等第排名值從高到低依序填入grade欄位(資料操弄時多採用邏輯值索引，避免使用if…then…條件式語法)
roster$grade[score >= y[1]] <- "A"
roster$grade[score < y[1] & score >= y[2]] <- "B"
roster$grade[score < y[2] & score >= y[3]] <- "C"
roster$grade[score < y[3] & score >= y[4]] <- "D"
roster$grade[score < y[4]] <- "F"

## ------------------------------------------------------------------------
# 以空白斷開名與姓
name <- strsplit(roster$Student, " ")
name[1:2]
# sapply()取出串列name的一個個向量元素後再結合中括弧取值運算子
(lastname <- sapply(name, "[", 2))
#同前，取向量的第一個元素
(firstname <- sapply(name, "[", 1))
# 資料物件取值函數說明頁面
# ?"["

## ------------------------------------------------------------------------
# 移除原Student欄位，添加firstname與lastname兩欄位
roster <- cbind(firstname, lastname, roster[,-1])
# order()依傳入的字串向量lastname之字詞升冪順序傳出觀測值編號
# 平手時依第二個向量firstname決定順序
(lexiAscen <- order(lastname, firstname))
# R語言常用的表格排序做法
roster <- roster[lexiAscen,]
# 注意列首的觀測值編號
roster

## ------------------------------------------------------------------------
# 簡例說明排序相關函數的區別，隨機產生10個均勻分配亂數
(x <- runif(10, min = 0, max = 1))
# 值進，排序後的索引值出(values in, sorted indices out)
order(x)
# 值進，排序後的值出(values in, sorted values out)
sort(x)
# sort()也可以傳回排序後的索引值
sort(x, index.return = TRUE)
# 值進，各元素排名值出(values in,ranks for each position out)
rank(x)

## 2.1.3 R語言資料變形 ------------------------------------------------------------------------
fname <- './_data/nst-est2015-popchg2010_2015.csv'
pop <- read.csv(fname)

## ------------------------------------------------------------------------
# 選取四個欄位
pop <- pop[,c("NAME","POPESTIMATE2010","POPESTIMATE2011",
"POPESTIMATE2012")]
# 簡化欄位名稱
colnames(pop) <- c('state', seq(2010, 2012))
head(pop, 6)

## ----warning=FALSE, message=FALSE----------------------------------------
library(reshape2)
# 寬(pop)轉長(mpop)
mpop <- melt(pop, id.vars = 'state', variable.name = 'year',
value.name = 'population')
# 適合變異數分析、ggplot2繪圖與資料庫儲存的長資料
head(mpop)

## ----eval=FALSE----------------------------------------------------------
## # state是州名、year是西元年、population是人口估計值
## str(mpop)

## ------------------------------------------------------------------------
# 長轉寬，順序與前面pop不同
dcast(mpop, state~year, value.var = 'population')[1:5,]

## ----warning=FALSE, message=FALSE----------------------------------------
library(tidyr)
library(dplyr)
# 放大dplyr橫向顯示寬度
options(dplyr.width = Inf)
billboard <- read.csv('./_data/billboard.csv',
stringsAsFactors = FALSE)
# names(billboard) # 最長追蹤76週的排名("x76th.week")
dim(billboard)
# billboard資料集變數眾多，礙於篇幅，只檢視前九個
head(billboard[1:9])

## ------------------------------------------------------------------------
# 寬表收起(gather)為長表
# 管路運算子語法同billboard2 <- gather(billboard, key = week,
# value = rank, x1st.week:x76th.week)
# 從x1st.week到x76th.week收集成key引數指名的week
# 欄位，各週對應的排名值收集成value引數指名的rank欄位
billboard2 <- billboard %>% gather(key = week, value = rank,
x1st.week:x76th.week)
headtail(billboard2, 3)

## ------------------------------------------------------------------------
# 長表散開(spread)成寬表
billboard3 <- billboard2 %>% spread(week, rank)
# 管路運算子語法同billboard3 <-spread(billboard2, week, rank)
# 結果與billboard相同，只是欄位順序不一樣
head(billboard3[1:9])


## 2.1.5 R語言資料清理 -----統計函數大多不能接受遺缺值！-------------------------------------------------------------------
x <- c(1, 2, 3, NA)
# 向量元素加總產生NA
(y <- x[1] + x[2] + x[3] + x[4])
# 加總函數的結果也是NA
(z <- sum(x))
# 移除NA後再做加總計算
(z <- sum(x, na.rm = TRUE))

## ------------------------------------------------------------------------
# 遺缺值NA辨識函數
is.na(x)

# 取得遺缺值位置編號(Which one is TRUE?)
which(is.na(x))

## ------------------------------------------------------------------------
manager <- c(1, 2, 3, 4, 5)
date <- c("10/24/08", "10/28/08", "10/1/08", "10/12/08",
"5/1/09")
country <- c("US", "US", "UK", "UK", "UK")
gender <- c("M", "F", "F", "M", "F")
age <- c(32, 45, 25, 39, 99)
q1 <- c(5, 3, 3, 3, 2)
q2 <- c(4, 5, 5, 3, 2)
q3 <- c(5, 2, 5, 4, 1)
q4 <- c(5, 5, 5, NA, 2)
q5 <- c(5, 5, 2, NA, 1)
(leadership <- data.frame(manager, date, country, gender,
age, q1, q2, q3, q4, q5, stringsAsFactors = FALSE))
# 二維資料表遺缺值辨識
is.na(leadership[,6:10])
# 橫向移除有遺缺值的觀測值
newdata <- na.omit(leadership)
# 第4筆觀測值被移除因此跳號
newdata
# 可以重新設定橫向索引rownames為流水號(optional)
rownames(newdata) <- 1:nrow(newdata)
newdata
# 本節running example ####
## ----warning = FALSE, message=FALSE--------------------------------------
# 載入R套件與資料集
library(DMwR) # library(DMwR2)
data(algae)

## ----eval=FALSE----------------------------------------------------------
## str(algae)

## ------------------------------------------------------------------------
# is.na()傳回200個是否遺缺的真假值(結果未全部顯示，後不贅述)
is.na(algae$mxPH)[1:48]
# 合成函數語法，快速知曉遺缺位置
which(is.na(algae$mxPH))
# 直接移除NA並另存為mxPH.na.omit
mxPH.na.omit <- na.omit(algae$mxPH)
length(mxPH.na.omit)
# 說明遺缺值處理方式的詮釋資料
attributes(mxPH.na.omit)

## ------------------------------------------------------------------------
# 有NA就報錯的處理方式na.fail()
# na.fail(algae$mxPH)
# Error in na.fail.default(algae$mxPH) : missing values in
# object

## ----warning = FALSE, message=FALSE--------------------------------------
# R語言多重補值套件{mice}
library(mice)
# 各種遺缺型態統計報表
md.pattern(algae, plot = TRUE) # 任何圖形可能都源自表格！

## ----fig.align="center", fig.cap = "\\label{fig:algae_aggr}水質樣本遺缺型態視覺化圖形", warning = FALSE, message=FALSE----
# R語言遺缺值視覺化與填補套件{VIM}
library(VIM)
aggr(algae, prop = FALSE, numbers = TRUE, cex.axis = .5) # 長條圖 + 熱圖/長條圖

naInfo <- is.na(algae)

rowSums(naInfo)
colSums(naInfo)

## ------------------------------------------------------------------------
# 各樣本(橫向)是否完整無缺
complete.cases(algae)[1:60]

## ------------------------------------------------------------------------
# 邏輯否定運算子搭配which()函數，抓出不完整樣本位置
which(!complete.cases(algae))

## ------------------------------------------------------------------------
# 取出不完整的樣本加以檢視
algae[which(!complete.cases(algae)),]
# 也可以用邏輯值索引取出不完整的樣本(請自行練習)
# algae[!complete.cases(algae),]

## ------------------------------------------------------------------------
# 以邏輯值索引移除不完整的觀測值
algae1 <- algae[complete.cases(algae),]

## ------------------------------------------------------------------------
# 統計各樣本遺缺變數個數
apply(algae, MARGIN = 1, FUN = function(x) {sum(is.na(x))})

## ------------------------------------------------------------------------
# 結果與complete.cases()一致
which(apply(algae, MARGIN = 1, FUN = function(x)
{sum(is.na(x))}) > 0)

## ------------------------------------------------------------------------
data(algae)
# 返回遺缺變數數量超過20%以上(nORp=0.2)的樣本編號
manyNAs(algae, nORp = 0.2)
# 負索引刪除遺缺程度較嚴重的樣本
# Python語言以DataFrame之drop()方法刪除
algae <- algae[-manyNAs(algae),]

## ----fig.align="center", fig.cap = "\\label{fig:algae_mxPH}最大酸鹼值mxPH散佈狀況圖", warning = FALSE, message=FALSE----
library(car) # J. Fox and S. Weisberg, An R Companion to Applied Regression, Third Edition, Sage, 2019.
# 圖面切分一列兩行(mfrow=c(1,2))，cex.main主標題文字縮小70%
par(mfrow = c(1, 2), cex.main = 0.7)
# 左行高階繪圖(直方圖)
hist(algae$mxPH, prob = T, xlab = '', main =
'Histogram of maximum pH value', ylim = 0:1)
# 左行低階繪圖兩次(密度曲線加一維散佈刻度)
lines(density(algae$mxPH, na.rm = T))
rug(jitter(algae$mxPH))
# 右行高階繪圖(常態機率繪圖，點靠近斜直線表近似常態分佈)
qqPlot(algae$mxPH, main = 'Normal QQ plot of maximum pH')
# 還原圖面一列一行原始設定
par(mfrow = c(1,1))

## ------------------------------------------------------------------------
# 用自己的算術平均數填補遺缺值
algae[48,'mxPH'] <- mean(algae$mxPH, na.rm = T)

## ----fig.align="center", fig.cap = "\\label{fig:algae_Chla}葉綠素Chla散佈狀況圖", warning = FALSE, message=FALSE----
par(mfrow = c(1,2))
hist(algae$Chla, prob = T, xlab='', main='Histogram of Chla')
lines(density(algae$Chla,na.rm = T))
rug(jitter(algae$Chla))
# 順帶返回偏離嚴重的樣本編號
qqPlot(algae$Chla,main = 'Normal QQ plot of Chla')
par(mfrow = c(1,1))
# 用自己的中位數填補遺缺值
algae[is.na(algae$Chla),'Chla'] <-
median(algae$Chla,na.rm = T)
# median(algae$Chla)
## ------------------------------------------------------------------------
data(algae)
# 移除遺缺狀況嚴重的樣本
algae <- algae[-manyNAs(algae),]
# 檢視遺缺狀況較不嚴重的樣本
algae[!complete.cases(algae),]
# 用各欄位自身的集中趨勢資訊進行填補
algae <- centralImputation(algae)
# 已無不完整的樣本了！
algae[!complete.cases(algae),]

## ------------------------------------------------------------------------
# Harrell Miscellaneous Functions套件
library(Hmisc)
data(algae)
# 以算術平均數填補mxPH遺缺值，星號顯示填補位置
impute(algae$mxPH, fun = mean)[40:55]
# Chla有遺缺值
summary(algae$Chla)
# 以中位數填補Chla遺缺值
impute(algae$Chla, fun = median)[50:65]
# 以固定數值45填補Chla遺缺值
impute(algae$Chla, fun = 45)[50:65]
# 以隨機產生的數值填補Chla遺缺值
impute(algae$Chla, fun = "random")[50:65]

## ------------------------------------------------------------------------
# "complete.obs"選項使用完整觀測值計算兩兩變數的相關係數
cor(algae[,4:18], use = "complete.obs")[6:11,6:11]
# "everything"用全部的觀測值計算相關係數，可能返回NA值
cor(algae[,4:18], use = "everything")[6:11,6:11]
# "all.obs"選項當觀測值中有NAs時會返回錯誤訊息
# cor(algae[,4:18], use = "all.obs")
# Error in cor(algae[, 4:18], use = "all.obs") :
# missing observations in cov/cor

## ------------------------------------------------------------------------
# "pairwise.complete.obs"選項使用成對完整的觀測值計算相關係數
cor(algae[,4:18], use = "pairwise.complete.obs")[6:11,6:11]
# pairwise.complete.obs與complete.obs兩者計算結果不完全相同！
cor(algae[,4:18], use = "pairwise.complete.obs") ==
cor(algae[,4:18], use = "complete.obs")
# 只有對角線上的相關係數值相同，其它全部不同！
sum(cor(algae[,4:18], use = "pairwise.complete.obs") ==
cor(algae[,4:18], use = "complete.obs"))

## ------------------------------------------------------------------------
# 相關係數矩陣符號化，*表PO4與oPO4係數絕對值超過0.9
symnum(cor(algae[,4:18],use = "complete.obs"))

## ------------------------------------------------------------------------
data(algae)
algae <- algae[-manyNAs(algae),]
# R語言線性建模重要函數lm()
(mdl <- lm(PO4 ~ oPO4, data = algae))

## ------------------------------------------------------------------------
(algae[28,'PO4'] <- 42.897 + 1.293 * algae[28,'oPO4'])

## ------------------------------------------------------------------------
data(algae)
algae <- algae[-manyNAs(algae),]
# 創造多個PO4遺缺的情境
algae$PO4[29:33] <- NA

## ------------------------------------------------------------------------
# 考慮連自變數oPO4都遺缺的邊界案例(edge case)(參見1.9節)
algae$oPO4[33] <- NA

## ------------------------------------------------------------------------
algae[is.na(algae$PO4), c('oPO4','PO4')]

## ------------------------------------------------------------------------
fillPO4 <- function(oP) { # oP即為自變量oPO4
  # 邊界案例處理
  if (is.na(oP)) return(NA)
  # 從模型物件mdl中取出迴歸係數進行補值計算
  else return(mdl$coef[1] + mdl$coef[2] * oP)
}

## ------------------------------------------------------------------------
# 邏輯值索引、隱式迴圈與自訂函數
algae[is.na(algae$PO4),'PO4']<-sapply(algae[is.na(algae$PO4),
'oPO4'], fillPO4)
# sapply(1:3, function(x) x + 2)
# 檢視填補完成狀況
algae[28:33, c('PO4', 'oPO4')]

## ------------------------------------------------------------------------
data(algae)
algae <- algae[-manyNAs(algae),]
# k近鄰填補函數，預設meth引數為加權平均"weighAvg"
algae <- knnImputation(algae, k = 10)

## ------------------------------------------------------------------------
data(algae)
algae <- algae[-manyNAs(algae),]
# 以近鄰的中位數填補遺缺值
algae <- knnImputation(algae,k=10, meth='median')

## ----fig.align="center", fig.cap = "\\label{fig:algae_mxPH_season}不同季節下最大酸鹼值mxPH的散佈狀況"----
data(algae)
algae <- algae[-manyNAs(algae),]
# 重要的建議套件
library(lattice)
# 更改默認的因子水準順序(預設是照字母順序)
algae$season <- factor(algae$season,levels =
c('spring','summer','autumn','winter'))
# mxPH條件式直方圖，依季節分層
histogram(~ mxPH | season, data = algae)

## ----fig.align="center", fig.cap = "\\label{fig:algae_mxPH_size}不同河流大小下最大酸鹼值mxPH的散佈狀況"----
# mxPH條件式直方圖，依河流大小分層
histogram(~ mxPH | size, data = algae)

## ------------------------------------------------------------------------
# mxPH遺缺該筆樣本的size為small
algae[is.na(algae$mxPH), 'size']
# 抓取size為small的所有樣本，計算平均mxPH值(顯然較低！)
mean(algae[algae$size == "small", 'mxPH'], na.rm = T)

## ----fig.align="center", fig.cap = "\\label{fig:algae_mxPH_size_speed}不同河流大小與流速下最大酸鹼值mxPH的散佈狀況"----
# 多因子變數的條件式直方圖，以*串接兩個分組變數
histogram(~ mxPH | size*speed, data = algae)

## ----fig.align="center", fig.cap = "\\label{fig:algae_mxPH_size_speed_strip}不同河流大小與流速下最大酸鹼值mxPH的點條圖"----
# 兩個因子變數，一個數值變數的點條圖
# 注意jitter=T是為了避免於同處過度繪製(overplotting)!
stripplot(size ~ mxPH | speed, data = algae, jitter = T)

## 2.2 資料摘要與彙總 ####
## 2.2.1 摘要統計量 ####
## ----warning=FALSE, message=FALSE----------------------------------------
library(nutshell)
data(dow30)
# dow30 <- read.csv('./About_nutshell_pkgs_Chapter2/dow30.csv', colClasses=c("NULL", rep(NA, 8)))
# ?as.POSIXct
## ----eval=FALSE----------------------------------------------------------
## # 股價資料框結構
## str(dow30)

## ------------------------------------------------------------------------
# 全體平均數(grand mean)
mean(dow30$Open)
# 截尾平均數
mean(dow30$Open, trim = 0.1) 

## ------------------------------------------------------------------------
# 也稱為第50個(50th)百分位數
median(dow30$Open) # try mean(dow30$Open, trim = 0.5)

## ----warning=FALSE, message=FALSE----------------------------------------
library(DMwR) # library(DMwR2)
data(algae)
# 自訂眾數計算函數
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}
Mode(algae$size)
# 以次數分佈表核驗Mode函數計算結果
table(algae$size)
unique(algae$size)[which.max(tabulate(match(algae$size, unique(algae$size))))] # 如何逐步理解程式碼
## ----warning = FALSE, message=FALSE--------------------------------------
mySamples <- c(19, 4, 5, 7, 29, 19, 29, 13, 25, 19) # c()想成combine!
# R語言單變量單峰資料眾數估計套件
library(modeest)
# 數值向量最常發生值估計法
mlv(mySamples, method = "mfv")
# table(mySamples)

## ------------------------------------------------------------------------
# 最常用的四分位數加上最小與最大值
quantile(dow30$Open, probs = c(0, 0.25, 0.5, 0.75, 1.0))

## ------------------------------------------------------------------------
# 結果與上面的四分位數有差異，其實quantile()函數有九種計算方法！
fivenum(dow30$Open)

## ------------------------------------------------------------------------
min(dow30$Open)
max(dow30$Open)

## ------------------------------------------------------------------------
# 一次傳回最小與最大值
range(dow30$Open)
# diff()常用於時間序列資料，依lag期數計算不同階數的差分值
diff(range(dow30$Open)) # 後 - 前
# 1到10跨兩期的8個(Why?)差分值
diff(1:10, lag = 2)
# R語言函數的引數可以偷懶只寫前幾個字母，只要能區辨即可
diff(1:10, lag = 2, diff = 2) # 跨兩期，差分運算兩次

## ------------------------------------------------------------------------
IQR(dow30$Open)

## ------------------------------------------------------------------------
var(dow30$Open) # 原始單位的平方
sd(dow30$Open) # 與原始單位一致
# 直接由標準差與平均數計算變異係數
sd(dow30$Open)/mean(dow30$Open)
# R語言地理資料分析與建模套件{raster}

## ----warning=FALSE, message=FALSE----------------------------------------
library(raster)

## ------------------------------------------------------------------------
# 光柵(raster)資料常須計算變異係數
cv(1:10)
sd(1:10)/mean(1:10)
# cv(dow30$Open) # cv*100
## ------------------------------------------------------------------------
# R核心開發團隊維護套件{stats}中的mad()
mad(dow30$Open) # mad: median absolute deviation 中位數絕對離差

## ----warning=FALSE, message=FALSE----------------------------------------
# 衡量所得不均、集結度與貧困的R套件
library(ineq)
# 建立所得向量x
x <- c(541, 1463, 2445, 3438, 4437, 5401, 6392, 8304,
11904, 22261)
# 吉尼集中度計算函數
ineq(x)

## ------------------------------------------------------------------------
# 自定義吉尼不純度函數
Gini <- function(x) {
  # 合理性檢查
  if (!is.factor(x)) {
    return("Please input factor variable.")
  } else {
    1 - sum((table(x)/sum(table(x)))^2) # (2.1)式
  }
}
# 完美同質情況
as.factor(rep("a", 6)) # "a"重複6次
Gini(as.factor(rep("a", 6)))

## ------------------------------------------------------------------------
# 非完美情況
as.factor(c(rep("a", 1), rep("b", 5)))
Gini(as.factor(c(rep("a", 1), rep("b", 5))))
# 完美異質情況
as.factor(c("a", "b", "c", "d", "e"))
Gini(as.factor(c("a", "b", "c", "d", "e"))) # 1 - 1/k = 1 - 1/5 = 0.8

## 2.2.2 R語言群組與摘要 ------------------------------------------------------------------------
# 知名的鳶尾花資料集
head(iris, 3) # iris.head(n=3) in Python

## ----results='hide'------------------------------------------------------
str(iris)

## ------------------------------------------------------------------------
# 量綱接近，花瓣寬度數值最小
summary(iris)

## ------------------------------------------------------------------------
# subset()選取各花種子集用法
setosa <- subset(iris, Species == 'setosa')
headtail(setosa)
# 逐步收納結果用(for gathering results)
results <- data.frame()
# 留意因子變數unique()後的結果
for (species in unique(iris$Species)) {
  # 逐花種取子集
  tmp <- subset(iris, Species == species)
  # 開始摘要統計
  count <- nrow(tmp)
  mean <- mean(tmp$Sepal.Length)
  median <- median(tmp$Sepal.Length)
  # 結果封裝成資料框後再合併
  results <- rbind(results, data.frame(species, count, mean,
  median))
}
results

## ------------------------------------------------------------------------
# 群組與摘要apply()系列函數
tapply(iris$Sepal.Length, iris$Species, FUN = length)
# FUN中設定的摘要統計計算函數，也可以用匿名函數定義多個函數
# 匿名函數的參數u代表分組數據
tapply(iris$Sepal.Length, iris$Species, FUN = function(u)
{c(count = length(u), meanUsr = mean(u), medianUsr = median(u))})

## ------------------------------------------------------------------------
# 限定環境與結合模型公式符號的aggregate()
aggregate(Sepal.Length ~ Species, data = iris,
FUN = 'length')
# 匿名函數用法同tapply()，但傳回結果為data.frame非list
aggregate(Sepal.Length ~ Species, data = iris, FUN =
function(u) {c(count = length(u), meanUsr = mean(u),
medianUsr = median(u))})

## ----warning=FALSE, message=FALSE----------------------------------------
# 載入R語言知名圖形文法繪圖套件{ggplot2}
library(ggplot2)
# 讀取套件{ggplot2}中的鑽石資料集
data(diamonds)
# 物件類別為"tbl_df"
class(diamonds)
head(diamonds)

## ----eval=FALSE----------------------------------------------------------
## # 注意cut與color的水準數
## str(diamonds)

## ----warning=FALSE, message=FALSE----------------------------------------
library(UsingR)
# 以模型公式符號的加號運算子串起兩個因子變數
headtail(aggregate(price ~ cut + color, data = diamonds,
FUN = "mean"))
# 以cbind()組織所有待分組變數
headtail(aggregate(cbind(price, carat) ~ cut + color,
data = diamonds, "mean"))
# 欲群組摘要的變數也可以是因子變數
headtail(aggregate(clarity ~ cut + color, data = diamonds,
"table"))
# clarity因子變數的各個水準名稱
levels(diamonds$clarity)

## ----warning=FALSE, message=FALSE----------------------------------------
library(doBy)
# 輸出格式與aggregate()相同
summaryBy(Sepal.Length ~ Species, data = iris, FUN =
function(x) {c(count = length(x), mean = mean(x), median =
median(x))})

## ----warning=FALSE, message=FALSE----------------------------------------
library(plyr)
# 留意傳入colMeans()的資料子集x為何要移除第五欄
ddply(iris, 'Species', function(x) c(count = nrow(x), mean
= colMeans(x[-5])))

## ----warning=FALSE, message=FALSE----------------------------------------
# R語言data.frame的延伸套件
library(data.table)
iris.tbl <- data.table(iris)
# 前者(data.table)繼承自後者(data.frame)
class(iris.tbl)
# list()串接多個變數進行取值，顯示方式自動取頭尾
iris.tbl[ , list(Sepal.Length, Species)]
# data.table橫列邏輯值取值
iris.tbl[iris.tbl$Petal.Width <= 0.1,]
# data.table移除變數是令其為NULL
iris.tbl[ , Sepal.Width := NULL]; iris.tbl

## ------------------------------------------------------------------------
# data.table群組與摘要語法特殊
iris.tbl[ , list(Sepal.Length = mean(Sepal.Length),
Petal.Width = median(Petal.Width)), by = Species]

## 2.3 屬性工程 ####
## 2.3.1 屬性轉換與移除 ------------------------------------------------------------------------
# R語言產生隨機抽樣模擬數據
set.seed(1234)
(X <- matrix(runif(12), 3))
# 橫向正規化(固定和)
apply(X, 1, sum)
(X_sum100 <- X/apply(X, 1, sum)*100)
# 核驗結果
apply(X_sum100, 1, sum)
# 橫向正規化(固定最大值)
apply(X, 1, max)
(X_max100 <- X/apply(X, 1, max)*100)
# 核驗結果
apply(X_max100, 1, max)
# 橫向正規化(固定向量長)
sqrt(apply(X^2, 1, sum))
(X_length1 <- X/sqrt(apply(X^2, 1, sum)))
# 核驗結果
apply(X_length1^2, 1, sum)

## ------------------------------------------------------------------------
# 本節 running example ####
library(AppliedPredictiveModeling)
data(segmentationOriginal)

## ----eval=FALSE----------------------------------------------------------
## str(segmentationOriginal, list.len = 10)

## ----warning=FALSE, message=FALSE----------------------------------------
library(UsingR)
# 119個變數名稱，限於篇幅，只顯示頭尾
headtail(names(segmentationOriginal))
# 目標變數`Class`次數分佈`table`(segmentationOriginal$Case)

## ----eval=FALSE----------------------------------------------------------
## # 報表太大，請讀者自行執行
## summary(segmentationOriginal)

## ------------------------------------------------------------------------
# 自變項資料框傳入sapply()，以匿名函數計算各變數最小與最大值
headtail(sapply(segmentationOriginal[-(1:3)], function(u)
c(min = min(u), max = max(u))))

## ------------------------------------------------------------------------
# 因子屬性`Case`次數分佈`table`(segmentationOriginal$Case)
segTrain <- subset(segmentationOriginal, Case == "Train")
# table(segmentationOriginal$Case)
## ------------------------------------------------------------------------
# 類別標籤向量獨立為segTrainClass，1009個類別標籤值
segTrainClass <- segTrain$Class
length(segTrainClass)

## ------------------------------------------------------------------------
segTrainX <- segTrain[, -(1:3)]
# 1009個訓練樣本，116個屬性
dim(segTrainX)

## ----warning=FALSE, message=FALSE----------------------------------------
# 分類與迴歸訓練重要R套件
library(caret)
# 預設傳回近乎零變異的變數編號
nearZeroVar(segTrainX)
# ?nearZeroVar
# 近乎零變異變數名稱
names(segTrainX)[nearZeroVar(segTrainX)]
# 移除近乎零變異變數後存為segTrainXV
segTrainXV <- segTrainX[, -nearZeroVar(segTrainX)]
#table(segTrainX$MemberAvgAvgIntenStatusCh2)
## ------------------------------------------------------------------------
# 改變引數(saveMetrics)設定值，取得完整報表(報表很長，只檢視某些變數結果)
nearZeroVar(segTrainX, saveMetrics = TRUE)[68:75,]
# 輸出報表類別為資料框
class(nearZeroVar(segTrainX, saveMetrics = TRUE))
str(nearZeroVar(segTrainX, saveMetrics = TRUE))
# 零變異變數名稱
names(segTrainX)[nearZeroVar(segTrainX, saveMetrics =
TRUE)$zeroVar]

## ------------------------------------------------------------------------
# 擷取名稱帶有"Status"的變數編號
(statusColNum <- grep("Status", names(segTrainXV)))

## ------------------------------------------------------------------------
# 名稱有"Status"之變數成批產生次數分配表，確定均為三元以下變數
head(sapply(segTrainXV[, statusColNum], table))
# 分離出數值屬性矩陣segTrainXNC，NC表not categorical
segTrainXNC <- segTrainXV[, -statusColNum]

## ------------------------------------------------------------------------
# VarIntenCh3嚴重右偏
summary(segTrainXNC$VarIntenCh3)
max(segTrainX$VarIntenCh3)/min(segTrainX$VarIntenCh3)

## ----warning=FALSE, message=FALSE----------------------------------------
# 偏態係數計算
library(e1071)
skewness(segTrainXNC$VarIntenCh3)

## ----fig.align="center", fig.cap = "\\label{fig:VarIntenCh3_histden}肌動蛋白絲畫素強度標準差之直方圖與密度曲線"----
# 視覺化檢驗偏態狀況
hist(segTrainXNC$VarIntenCh3, prob = TRUE, ylim =
c(0, 0.009), xlab = 'VarIntenCh3')
lines(density(segTrainXNC$VarIntenCh3))
# 以位置量數細部查驗偏態狀況(注意66%與97%)
quantile(segTrainXNC$VarIntenCh3, probs = seq(0, 1, 0.01))

## ------------------------------------------------------------------------
# 以apply()+sort()計算並排序所有數值變數的偏態係數
skewValues <- apply(segTrainXNC, 2, skewness)
headtail(sort(skewValues, decreasing = TRUE))

## ----fig.align="center", fig.cap = "\\label{fig:rightskewed_hist}右偏前九高變數的直方圖", fig.height=8----
# 偏態係數高於3的變數
sort(skewValues[skewValues > 3], decreasing = TRUE)
# 取出右偏前九高的變數名稱
highlySkewed <- names(sort(skewValues, decreasing=TRUE))[1:9]
# 成批繪製直方圖
op <- par(mfrow = c(3,3))
invisible(lapply(highlySkewed, function(u)
{hist(segTrainXNC[[u]], main = paste("Histogram of ",
u), xlab = "", cex.main = 0.8)}))
par(op)

## ----echo=FALSE, fig.align="center", fig.cap = "\\label{fig:leftskewed_hist}左偏前九高變數的直方圖", fig.height=8----
symm <- names(sort(skewValues))[1:9]
op <- par(mfrow = c(3,3))
invisible(lapply(symm, function(u) {hist(segTrainXNC[[u]],
main = paste("Histogram of ", u),
xlab = "", cex.main = 0.8)}))
par(op)

## ----warning=FALSE, message=FALSE----------------------------------------
library(caret)
# BC轉換報表(lambda估計值為-0.9)
(Ch1AreaTrans <- BoxCoxTrans(segTrainXNC$AreaCh1))

## ------------------------------------------------------------------------
# 前六筆原始資料
head(segTrainXNC$AreaCh1)
# predict()泛型函數執行BC轉換計算
predict(Ch1AreaTrans, head(segTrainXNC$AreaCh1))
# 自撰BC公式驗證
(head(segTrainXNC$AreaCh1)^(-.9) - 1)/(-.9)

## 2.3.2 屬性萃取之主成份分析 ------------------------------------------------------------------------
# 資料標準化後進行PCA運算
pcaObject <- prcomp(segTrainXNC, center = TRUE, scale. = TRUE)
# pcaObject <- prcomp(segTrainXNC, center = TRUE)
# pcaObject <- princomp(segTrainXNC)
# names(pcaObject) # "sdev"     "loadings" "center"   "scale"    "n.obs"    "scores"   "call"
# 確認主成份之間是否獨立
# tmp <- cor(pcaObject$scores)
# sum(tmp > 0.0001) # 96 !?
# sum(tmp > 0.01) # 80 !?
# sum(tmp > 0.1) # 68 !?

## ------------------------------------------------------------------------
# 檢視計算結果
names(pcaObject) # "sdev"     "rotation" "center"   "scale"    "x" (similar to dir(modelObject) in Python)

# 確認主成份之間是否獨立
# tmp <- cor(pcaObject$x)
# sum(tmp > 0.0001) # 128 !?
# sum(tmp > 0.01) # 104 !?
# sum(tmp > 0.05) # 64 !?
# sum(tmp > 0.1) # 58 !?

## ------------------------------------------------------------------------
# 驗證分數矩陣的計算
sum(scale(as.matrix(segTrainXNC)) %*% pcaObject$rotation ==
pcaObject$x)
# 分數矩陣中元素總數為樣本數乘上變數個數
1009*58

## ------------------------------------------------------------------------
# 各個主成份變異數占總變異數的比例逐漸遞減
(percentVariance <- pcaObject$sd^2/sum(pcaObject$sd^2)*100)

## ----fig.align="center", fig.cap = "\\label{fig:cell_screeplot}前25個主成份的陡坡圖"----
# 繪製陡坡圖
plot(percentVariance[1:25], xlab = "Principal Component",
ylab = "Proportion of Variance Explained ", type = 'b')

## ----fig.align="center", fig.cap = "\\label{fig:cell_cumvarplot}前25個主成份累積變異百分比折線圖"----
# 累積變異百分比折線圖
cumsum(percentVariance)
plot(cumsum(percentVariance)[1:25], xlab = "Principal
Component", ylab = "Cumulative Proportion of
Variance Explained", type = 'b')
abline(h = 0.9, lty = 2)

## ------------------------------------------------------------------------
# PCA座標轉換後的新座標值儲存在pcaObject中的元素x
head(pcaObject$x[, 1:5])

## ------------------------------------------------------------------------
# 檢視PCA前三個主成份與頭六個預測變數的負荷係數值
head(pcaObject$rotation[, 1:3])

## ------------------------------------------------------------------------
# preProcess()可以處理各種轉換，包括PCA
(segPP <- preProcess(segTrainXNC, method = c("BoxCox",
"center", "scale", "pca")))
# 物件類別為"preProcess"，通常與創建物件的函數同名
class(segPP)

## ------------------------------------------------------------------------
# 套用模型segPP對數值屬性矩陣做轉換
transformed <- predict(segPP, segTrainXNC)
# 原資料已降到19維的正交新空間中
str(transformed)
# 新空間前五個主成份下，前六筆樣本的座標值
head(transformed[, 1:5])
# predcit()方法轉換後輸出的物件類別為data.frame
class(transformed)


## 2.3.3 屬性挑選 ------------------------------------------------------------------------
# 58個數值變數的相關係數方陣
correlations <- cor(segTrainXNC)
dim(correlations)
correlations[1:4, 1:4]

## ----fig.align="center", fig.cap = "\\label{fig:cell_corrplot}相關係數矩陣視覺化圖形", warning=FALSE, message=FALSE, fig.height=6, fig.width=5----
# 載入R語言好用的相關係數方陣視覺化套件
library(corrplot)
corrplot(correlations, order = "hclust", tl.cex = 0.5)

# library(heatmaply)
# heatmaply_cor(correlations)

## ------------------------------------------------------------------------
# 傳回建議移除的變數編號
(highCorr <- findCorrelation(correlations, cutoff = .75))
# 原58個數值變數過濾掉32個剩下26個
filteredSegTrainXNC <- segTrainXNC[, -highCorr]
dim(filteredSegTrainXNC)

## ----warning=FALSE, message=FALSE----------------------------------------
# R語言屬性挑選套件
library(FSelector)
data(iris)
# 屬性訊息增益排名
(weights <- information.gain(Species~., data=iris))
(subset <- cutoff.k(weights, 2))
# 屬性訊息增益率排名
(weights <- gain.ratio(Species~., iris))
(subset <- cutoff.k(weights, 2))

## ----warning=FALSE, message=FALSE----------------------------------------
# 美國國會投票記錄資料集HouseVotes屬於套件{mlbench}下
library(mlbench)
data(HouseVotes84)
# help("HouseVotes84")

## ----eval=FALSE----------------------------------------------------------
## # 兩黨435位議員對16個法案的支持與否結果
## str(HouseVotes84)

## ------------------------------------------------------------------------
# 一致性指標選取10個屬性形成的最優子集(最具區辨兩黨投票行為的能力)
(subset <- consistency(Class~., HouseVotes84))

## 2.4 巨量資料處理概念 ####
## 2.4.1 文字資料處理 ------------------------------------------------------------------------
# 讀入三國章回小說片段內容，留意引數stringsAsFactors的設定值
text <- read.table("./_data/3KDText_Mac_short.txt",
stringsAsFactors = F) # Windows請讀取3KDText_Windows.txt，儲存成單列單欄的2D資料框
# class(text) # "data.frame"
# 強制轉換為字串向量(1D)
text <- as.character(text)

## ----warning=FALSE, message=FALSE----------------------------------------
library(jiebaR)
# 宣告(初始化)分詞器
seg <- worker()
# 傳入字串進行分詞，注意！text不可為資料框物件！
words <- seg <= text
words[1:50]
# {jiebaR}另一種分詞語法，結果同上
words <- segment(text, seg)

## ------------------------------------------------------------------------
# 加入使用者定義的新詞
new_user_word(seg, c("張寶", "張角"))
# 再次分詞，結果有斷出兩兄弟正確的名字
words <- seg <= text
words[1:50]

## ----eval=FALSE----------------------------------------------------------
## # {jiebaR}系統字典、使用者字典等預設設定內容(PrivateVarible)
## # 報表過寬，請讀者自行執行下行程式碼
## str(seg$PrivateVarible)

## ------------------------------------------------------------------------
# 關鍵字提取分詞器以混合分詞模型分詞後，再依詞頻-逆文件頻率篩選
# 前100個關鍵字
keys = worker(type = "keywords", topn = 100)
keywords <- keys <= text
keywords[1:10]

## ------------------------------------------------------------------------
# 宣告分詞類型為詞性標記的分詞器
tagger = worker(type = "tag")
tagCN <- tagger <= text
tagCN[1:10]
# 詞性標記結果tagCN為具名向量
(head(tag <- names(tagCN)))

## ----warning=FALSE, message=FALSE----------------------------------------
# 載入R語言英語自然語言處理套件
library(NLP)
library(openNLP)
# 建立練習語句
s <- c("Pierre Vinken, 61 years old, will join the board
as a nonexecutive director Nov. 29. Mr. Vinken is chairman
of Elsevier N.V., the Dutch publishing group.")

## ------------------------------------------------------------------------
# 練習語句字元數
nchar(s)
# 轉為{NLP}套件接受的物件類型String
s <- as.String(s)

## ------------------------------------------------------------------------
# 宣告斷句註記器
sent_token_annotator <- Maxent_Sent_Token_Annotator()
# 將英文語句放入後，透過先前宣告的sent_token_annotator函數
# 做斷句的動作
(a1 <- annotate(s = s, f = sent_token_annotator))
# 斷出兩個句子，注意start與end對應的字符位置編號
str(a1)

## ------------------------------------------------------------------------
# substr()取出第一個句子
substr(s, 1, 84)
# substr()取出第二個句子
substr(s, 86, 153)

## ----eval=FALSE----------------------------------------------------------
## # 直接將分句註記結果傳入String物件取值
## s[a1]

## ------------------------------------------------------------------------
# 加註成句的可能性
annotate(s, Maxent_Sent_Token_Annotator(probs = TRUE))

## ------------------------------------------------------------------------
# 宣告斷字註記器
word_token_annotator <- Maxent_Word_Token_Annotator()
# 傳入語句斷字，注意需接續前面斷句結果a1往下做
(a2 <- annotate(s = s, f = word_token_annotator, a = a1))
# 斷詞結果
tail(s[a2])

## ------------------------------------------------------------------------
# a2 <- annotate(s=s, f=word_token_annotator)
# Error in f(s, a) : no sentence token annotations found,
# an annotation object to start with must be given

## ------------------------------------------------------------------------
# 斷句做完接續做斷詞註解
a <- annotate(s, list(sent_token_annotator,
word_token_annotator))
# 結果同分段做一樣
head(a)

## ------------------------------------------------------------------------
# 宣告詞性標記註記器
# install.packages("openNLPmodels.en", dependencies=TRUE, repos = "http://datacube.wu.ac.at/") # 直接從本機硬碟安裝openNLPmodels.en_1.5-1.tar.gz或是install.packages("openNLPmodels.en", repos = "http://datacube.wu.ac.at/", type = "source")
pos_tag_annotator <- Maxent_POS_Tag_Annotator()
# 傳入語句標記詞性，注意需接續前面斷字結果a2往下做
(a3 <- annotate(s, pos_tag_annotator, a2))
# id = 1 & 2 被移掉了！因為type設定為"word"
(head(a3w <- subset(a3, type == "word")))

## ------------------------------------------------------------------------
# 宣告專有名詞註記器
entity_annotator <- Maxent_Entity_Annotator()
# 辨識出人名專有名詞(features: kind=person)
tail(annotate(s, entity_annotator, a2))
# 僅傳回專有名詞辨識結果
entity_annotator(s, a2)

## ------------------------------------------------------------------------
# 宣告詞組註記器
chunk_annotator <- Maxent_Chunk_Annotator()
# 詞組辨識結果
head(chunk_annotator(s, a3))
# 取出第一組B-NP與I-NP標籤的內容
s[chunk_annotator(s, a3)][1:2]
# 確認下一個標籤O的內容是無關的
s[chunk_annotator(s, a3)][3]

## ------------------------------------------------------------------------
# 分句、分詞、詞性標記與詞組提取的完整結果
chunk_annotator <- annotate(s = s, f =
Maxent_Chunk_Annotator(), a = a3)
head(chunk_annotator, 11)

