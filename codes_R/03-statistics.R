## Collated by Ching-Shih Tsou 鄒慶士 博士 (Ph.D.) Distinguished Prof. at the Department of Mechanical Engineering/Director at the Center of Artificial Intelligence & Data Science (機械工程系特聘教授兼人工智慧暨資料科學研究中心主任), MCUT (明志科技大學); Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), NTUB (國立臺北商業大學)
## Notes: This code is provided without warranty.

## 3.1 隨機誤差模型 ------------------------------------------------------------------------
library(DMwR2) # or library(DMwR2) for higher R version
data(algae)
algae$season <- factor(algae$season, levels = c("spring", "summer", "autumn", "winter"))
algae$size <- factor(algae$size, levels = c("small", "medium", "large"))
algae$speed <- factor(algae$speed, levels = c("low", "medium", "high"))

# 移除遺缺程度嚴重的樣本(如果使用古典統計方法，或者支援向量機及神經網絡，則須對遺缺數據進行清理; 迴歸樹/模型樹可以接受樣本中的缺失值！)
algae <- algae[-manyNAs(algae), ]
# 以近鄰的中位數填補遺缺值(knnImputation背後是一個程序)
cleanAlgae <- knnImputation(algae, k = 10)
# 確認資料表中已無遺缺值
sum(is.na(cleanAlgae))
# 設定亂數種子，使得結果可重製
set.seed(1234)
# 隨機保留50個樣本稍後進行模型測試
idx <- sample(nrow(cleanAlgae), 50) # cleanAlgae.shape[0] in Python
# 切割出訓練與測試樣本
train <- cleanAlgae[-idx, 1:12]
test <- cleanAlgae[idx, 1:12]
# 以148個訓練樣本估計函數關係
a1Lm <- lm(a1 ~ ., data = train)
# 擬合完成後運用模型a1Lm估計訓練樣本的a1有害藻類濃度
trainPred <- predict(a1Lm, train[-12]) # alLm.predict(train[-12]) in Python
# 模型a1Lm摘要報表
summary(a1Lm) # 除了標籤編碼，R建模時會自動對因子變量做虛擬編碼！alLm.summary()
# 訓練樣本的模型績效指標RMSE值(參見3.2.1節)
# 因為球員兼裁判，所以較為樂觀
(trainRMSE <- sqrt(mean((train$a1 - trainPred)^2)))
# 以模型a1Lm估計測試樣本的a1有害藻類濃度
testPred <- predict(a1Lm, test[-12])
# 測試樣本的模型績效指標RMSE值
(testRMSE <- sqrt(mean((test$a1 - testPred)^2)))
summary(algae$a1)

## ----echo = FALSE--------------------------------------------------------
# x <- .Random.seed
# save(x, file = "./_data/randomSeedHoldout.RData")
# load("./_data/randomSeedHoldout.RData") # sample可能改版，導致結果稍有不同！
# 參見./_data/下的random_seed.txt，將其中的一個數字替換到下方set.seed()函數中的亂數種子
set.seed(89)
idx <- sample(nrow(cleanAlgae), 50)
# 切割出訓練與測試樣本
train <- cleanAlgae[-idx, 1:12]
test <- cleanAlgae[idx, 1:12]
# 以148個訓練樣本估計函數關係
a1Lm2 <- lm(a1 ~ ., data = train)
# 擬合完成後運用模型a1Lm2估計訓練樣本的a1有害藻類濃度
trainPred <- predict(a1Lm2, train[-12])

## ----echo = FALSE, results = "hide"--------------------------------------
# 模型a1Lm2摘要報表
summary(a1Lm2)

## ------------------------------------------------------------------------
# 測試集的RMSE比訓練集RMSE還低的結果(實驗需要多做幾次！)
(trainRMSE <- sqrt(mean((train$a1 - trainPred)^2)))
testPred <- predict(a1Lm2, test[-12])
(testRMSE <- sqrt(mean((test$a1 - testPred)^2)))
summary(algae$a1)

## 3.2 模型績效評量 ####
## 3.2.1 迴歸模型績效指標 ####
## 3.2.2 分類模型績效指標 ####
## 3.2.2.1 模型預測值(類別標籤或陽性事件機率值) ####
## 3.2.2.2 混淆矩陣 ------------------------------------------------------------------------
# 相同亂數種子下結果可重置(reproducible)
set.seed(4321)
# 置回抽樣(replace引數)與設定各類被抽出的機率(prob引數)
observed <- sample(c("No", "Yes"), size = 30, replace = TRUE,
prob = c(2/3, 1/3))
# 觀測值向量一維次數分佈表
table(observed)
predicted<-sample(c("No", "Yes"), size = 30, replace = TRUE,
prob = c(2/3, 1/3))
# 預測值向量一維次數分佈表
table(predicted)
table(observed, predicted) # Two-way contingency table (二維列聯表) for the joint distribution


## ------------------------------------------------------------------------
# 混淆矩陣與正確率
(tbl <- table(observed, predicted))
#從混淆矩陣計算正確率
(acc <- (tbl[1,1] + tbl[2,2])/sum(tbl))
# 橫向邊際和，同觀測值次數分佈表
margin.table(tbl, margin=1)
# 縱向邊際和，同預測值次數分佈表
margin.table(tbl, margin=2)
# 期望正確率=(No橫縱邊際和乘積+Yes橫縱邊際和乘積)/樣本數平方
(exp <- (margin.table(tbl,1)[1]*margin.table(tbl,2)[1] +
margin.table(tbl,1)[2]*margin.table(tbl,2)[2])/(sum(tbl)^2))
# Kappa統計值
(kappa <- (acc - exp)/(1 - exp))

## 3.2.3 模型績效視覺化 ####
## ----warning=FALSE, message=FALSE, fig.align='center', fig.cap = "\\label{fig:rocplot}接收者操作特性圖"----
# 創建表3.1中的真實類別標籤與陽例機率預測值
actual <- factor(c(rep("p", 10), rep("n", 10)))
predProb <- c(0.9,0.8,0.6,0.55,0.54,0.51,0.4,0.38,0.34,0.3,
0.7,0.53,0.52,0.505,0.39,0.37,0.36,0.35,0.33,0.1)
ex <- data.frame(actual, predProb)
# 載入R語言ROC曲線繪製與分析套件
library(pROC)
# 建構曲線繪製的roc類別物件
rocEx <- roc(actual ~ predProb, data = ex)
class(rocEx)
# plot()方法繪圖，其實是呼叫plot.roc()方法
plot(rocEx, print.thres=TRUE, grid=TRUE, legacy.axes=TRUE)
# ?plot.roc
## ------------------------------------------------------------------------
# 傳入roc類別物件至AUC計算函數中
auc(rocEx)
# 或直接傳入真實標籤向量與陽例機率預測向量
auc(ex$actual, ex$predProb)

## 3.3 模型選擇與評定 ####
## 3.3.1 重抽樣與資料切分方法 ####
## ----fig.align="center", fig.cap = "\\label{fig:ozone_scatterplot}風速和溫度分別與臭氧的散佈圖"----
# 瞭解airquality的資料結構
str(airquality)

# 遺缺值查核與處理
# colSums(is.na(airquality))
# R為何無須遺缺值填補
# ?lm
# options('na.action')

# 分別視覺化Ozone與Wind和Temp的關係
op <- par(mfrow = c(1,2))
plot(Ozone ~ Wind, data = airquality,
main = "Ozone against Wind")
plot(Ozone ~ Temp, data = airquality,
main = "Ozone against Temp")
# 還原圖面原始設定
par(op)

## ----warning=FALSE, message=FALSE----------------------------------------
# R語言拔靴抽樣套件{boot}
library(boot)
# 定義拔靴抽樣函數boot()所用的統計值計算函數rsq()
rsq <- function(formula, data, indices) {
  # 結合下面boot()函數，選取拔靴樣本d
  d <- data[indices,]
  # 以拔靴樣本d建立模型
  fit <- lm(formula, data = d) # 此處未做遺缺值處理，但lm()中有na.action設定，請參照options('na.action')的說明
  # 返回模型適合度統計量r.square
  return(summary(fit)$r.square)
}

## ------------------------------------------------------------------------
# 請讀者自行檢閱boot()說明文件?boot，並留意引數關鍵字
# 拔靴抽樣建模與統計計算完成後存為bootaq物件
bootaq <- boot(data = airquality, statistic = rsq, R = 1000,
formula = Ozone ~ Wind + Temp)

# summary(lm(Ozone ~ Wind + Temp, data = airquality))$r.square # 0.5687097 ~= 0.569, t0

## ------------------------------------------------------------------------
# 1000次拔靴抽樣下迴歸模型的r.square值
head(bootaq$t)
length(bootaq$t)

## ----eval=FALSE----------------------------------------------------------
## # 拔靴抽樣統計報表
## bootaq

## ------------------------------------------------------------------------
class(bootaq)

## ----warning=FALSE, message=FALSE, fig.align="center", fig.cap = "\\label{fig:bootstrapplot}拔靴抽樣樣本統計量$R^{2}$直方圖與常態分位數圖"----
# 拔靴樣本統計量繪圖
plot(bootaq)
# r.square區間估計
boot.ci(bootaq)

## ----echo=FALSE, eval=FALSE----------------------------------------------
## names(bootaq)

## ------------------------------------------------------------------------
# 載入R語言套件與二元分類資料集
library(AppliedPredictiveModeling)
data(twoClassData)
# 檢視屬性矩陣結構
str(predictors)
# 檢視類別標籤因子向量結構
str(classes)

## ------------------------------------------------------------------------
library(caret)
#簡單保留法
trainingRows <- createDataPartition(classes, p = .80,
list=FALSE)
head(trainingRows)

## ------------------------------------------------------------------------
# 訓練集屬性矩陣與類別標籤
trainPredictors <- predictors[trainingRows, ]
trainClasses <- classes[trainingRows]
# 測試集屬性矩陣與類別標籤(R語言負索引值)
testPredictors <- predictors[-trainingRows, ]
testClasses <- classes[-trainingRows]
# 208*0.8無條件進位後取出167個訓練樣本
str(trainPredictors)
# 剩下41個測試樣本
str(testClasses)
length(testClasses)

## ------------------------------------------------------------------------
set.seed(1)
# 重複三次的保留法(又稱為LGOCV, Leave Group Out Cross-Validation)
repeatedSplits <- createDataPartition(trainClasses, p = .80,
times = 3)
# 三次訓練集樣本編號形成的串列
str(repeatedSplits)

## ------------------------------------------------------------------------
set.seed(1)
# 將校驗集樣本做十摺互斥切分(圖3.12為四摺)
cvSplits <- createFolds(trainClasses, k = 10, returnTrain
= TRUE)
# 返回十次測試樣本編號(returnTrain預設為FALSE)
cvSplitsTest <- createFolds(trainClasses, k = 10)
# 訓練樣本無法十等分
# 所以167/10 ~= 16.7, 每次結果為151 + 16或150 + 17
str(cvSplits)
# 各參數組合第一次交叉驗證的151個訓練樣本
cvSplits[[1]]
# 差集運算函數求取第一次交叉驗證的16個測試樣本
setdiff(1:167, cvSplits[[1]])

## 3.3.2 單類模型參數調校 ####
## ----warning=FALSE, message=FALSE----------------------------------------
# 載入R套件與資料集
library(ISLR) # Machine Learning -> Statistical Learning -> Probabilistic Machine Learning
library(caret)
data(Smarket)

# sum(is.na(Smarket)) # 0

# Smarket$Direction
# 驗證Smarket$Direction與Smarket$Today的關係
# table(Smarket$Year)
# temp <- ifelse(Smarket$Today >= 0, "Up", "Down")
# table(temp)
# table(Smarket$Direction)
# sum(temp == as.character(Smarket$Direction))
# Smarket$Direction[Smarket$Today > 0] # 647
# Smarket$Direction[Smarket$Today < 0] # 602
# Smarket$Direction[Smarket$Today == 0] # 1

## ------------------------------------------------------------------------
# 校驗集(75%)與(最終)測試集(25%)切分
set.seed(300)
indxCalib <- createDataPartition(y = Smarket$Direction,
p = 0.75,list = FALSE)
calibration <- Smarket[indxCalib, -1] # 註：此例將'Year'變量移除再進行kNN建模較為合理
testing <- Smarket[-indxCalib, -1]
# 類別分佈型態查核(校驗集vs.測試集vs.整體)
prop.table(table(calibration$Direction)) * 100
prop.table(table(testing$Direction)) * 100
prop.table(table(Smarket$Direction)) * 100

## ------------------------------------------------------------------------
# 校驗集屬性矩陣與前處理
calibX <- calibration[, names(calibration) != "Direction"]
preProcValues <- preProcess(x = calibX, method = c("center",
"scale"))
preProcValues

## ------------------------------------------------------------------------
set.seed(400)
# 校驗集重抽樣方法設定
ctrl <- trainControl(method = "repeatedcv", repeats = 3)
# 圖3.15對每一個參數候選值進行訓練與測試
knnFit <- train(Direction ~ ., data = calibration, method = # 注意！傳入calibration而非calibX！所以需要標準化
"knn", trControl = ctrl, preProcess = c("center","scale"),
tuneLength = 20) # 每個k值實驗了3*10(30)次後，再計算各次的正確率(Acccuracy)，然後統計30次的平均值(knnFit$results$Accuracy)與標準差(knnFit$results$AccuracySD)

## ----fig.align="center", fig.cap = "\\label{fig:knn_tuningplot}$k$近鄰法參數調校圖"----
# 重要的參數調校報表
knnFit
# 各參數候選值更詳細的績效評量概況
knnFit$results
# 圖勝於表，上表前兩欄的折線圖
plot(knnFit)

## ------------------------------------------------------------------------
# 自動以最佳模型(k=13原43)預測最終測試集樣本
knnPredict <- predict(knnFit, newdata = testing) # knnFit.predict(newdata = testing) in Python
# 混淆矩陣、整體分類績效指標與類別相關指標
confusionMatrix(knnPredict, testing$Direction)
# Smarket$Direction # So, the "Down" will be the "positive" result.

## ------------------------------------------------------------------------
# 核驗Accuracy是否正確
mean(knnPredict == testing$Direction)

## ----fig.align="center", fig.cap = "\\label{fig:knn_rocplot}$k$近鄰分類模型之測試資料操作特性曲線圖"----
# 載入ROC曲線繪製與分析套件
library(pROC)
# 繪製ROC曲線需計算測試資料的類別機率預測值
knnPredict <- predict(knnFit, newdata = testing, type = "prob")
knnROC <- roc(testing$Direction,knnPredict[,"Down"]) # 注意陽性事件定義為"Down"
# 類別"roc"
class(knnROC)
# AUC值
knnROC$auc
# 繪製ROC曲線
plot(knnROC, type = "S", print.thres = 0.5)

save(knnFit, file = 'knnFit_mac.RData')

## 3.3.2.1 多個參數待調 ------------------------------------------------------------------------
# 匯入信用風險資料集
credit <- read.csv("./_data/credit.csv")

## ----eval=FALSE----------------------------------------------------------
## # 變數意義如上
## str(credit)

## ----warning=FALSE, message=FALSE----------------------------------------
library(caret)
# C5.0決策樹自動參數調校
set.seed(300)
# 預設為重複25次的拔靴抽樣法，結果存為C50fit.RData
# C50fit <- train(default ~ ., data = credit, method =
# "C5.0") # 請檢視trainControl()
# save(C50fit, file = "C50fit.RData")
# 因模型建立與調校耗時，載入預先跑好的模型物件
load("./_data/C50fit.RData")

## ----eval=FALSE----------------------------------------------------------
# 多參數調校報表，讀者可自行檢視其結構str(C50fit)
C50fit

## 3.3.2.2 客製化參數調校 ------------------------------------------------------------------------
# 自訂校驗集重抽樣策略與擇優方法
ctrl <- trainControl(method = "cv", number = 10,
selectionFunction = "oneSE") # selectionFunction預設值為"best"
# 自訂待調參數組合(網格)
grid <- expand.grid(.model = "tree", .trials = c(1, 5, 10,
15, 20, 25, 30, 35), .winnow = "FALSE")
grid

## ------------------------------------------------------------------------
set.seed(300)
C50fitC <- train(default ~ ., data = credit, method =
"C5.0", metric = "Kappa", trControl = ctrl,
tuneGrid = grid) # metric預設值為ifelse(is.factor(y), "Accuracy", "RMSE")
# save(C50fitC, file = "C50fitC.RData")
# 因模型建立與調校耗時，載入預先跑好的模型物件
load("./_data/C50fitC.RData")
# 參數調校報表
C50fitC
# 各參數候選值更詳細的績效評量概況
C50fitC$results

plot(C50fitC)

## 3.3.3 比較不同類的模型 ------------------------------------------------------------------------
set.seed(1056)
# 羅吉斯迴歸建模方法為glm廣義線性模型
logisticReg <- train(default ~ ., data = credit, method =
"glm", preProc = c("center", "scale"))
# save(logisticReg, file = "logisticReg.RData")
# 因模型建立耗時，載入預先跑好的模型物件
load("./_data/logisticReg.RData")

## ----eval=FALSE----------------------------------------------------------
## # 25次拔靴抽樣下的羅吉斯迴歸建模報表
logisticReg

## ------------------------------------------------------------------------
# 跨模比較函數resamples()
resamp <- resamples(list(C50 = C50fit, Logistic
= logisticReg))
# 比較結果摘要報表
# Accuracy似乎C50佔上風，Kappa卻是Logistic全面勝出
summary(resamp)

par(mfrow=c(1,2))
boxplot(resamp$values$`C50~Accuracy`, xlab='C50', ylab='Accuracy')
boxplot(resamp$values$`Logistic~Accuracy`, xlab='LogisticReg', ylab='Accuracy')
par(mfrow=c(1,1))

## ------------------------------------------------------------------------
# resamples類別物件
class(resamp)
# 跨模型差異統計檢定
modelDifferences <- diff(resamp)
# 假說檢定(H0:無差異)摘要報表
summary(modelDifferences)

## ----fig.align="center", fig.cap = "\\label{fig:iris_scatterplot_mtx}三種鳶尾花花萼花瓣長寬成對散佈圖"----
# 鳶尾花資料集量化變數成對散佈圖
pairs(iris[1:4], main = "Anderson's Iris Data", pch = 21,
bg = c("red", "green3", "blue")[as.integer(iris$Species)])

## 3.5 相關與獨立 ####
## 3.5.1 數值變數與順序尺度類別變數 ####
## ----fig.align="center", fig.cap = "\\label{fig:smoothScatter}母親懷孕體重增加值與嬰兒體重關係圖(平滑散佈圖)"----
# 載入美國2006年新生兒資料集
library(nutshell)
# 取出WTGAIN與DBWT均無遺缺值的樣本子集
data(births2006.smpl)
births2006.cln <- births2006.smpl[
!is.na(births2006.smpl$WTGAIN) &
!is.na(births2006.smpl$DBWT) &
# 鎖定單胞胎與移除早產兒樣本
births2006.smpl$DPLURAL == "1 Single" &
births2006.smpl$ESTGEST > 35,]
dim(births2006.cln)
# R語言基礎繪圖套件{graphics}中的平滑散佈圖函數
smoothScatter(births2006.cln$WTGAIN, births2006.cln$DBWT,
xlab = "Mother's Weight Gained", ylab = "Baby Birth Weight")
# cor() 函數預設為皮爾森相關係數
cor(births2006.cln$WTGAIN, births2006.cln$DBWT)

## ------------------------------------------------------------------------
# 史皮爾曼相關係數
cor(births2006.cln$WTGAIN, births2006.cln$DBWT,
method = "spearman")

## ----warning=FALSE, message=FALSE----------------------------------------
# 模擬十筆5個變量的資料矩陣
(X <- matrix(runif(50, 1, 7), 10))
# 共變異數方陣
cov(X)
# 載入R語言穩健統計方法套件
library(robustbase)

## ----eval=FALSE----------------------------------------------------------
## # 最小共變異數判別式估計法，alpha即為前述之k
## covMcd(X, alpha = 0.75)

## 3.5.2 名目尺度類別變數 ####
## ----warning=FALSE, message=FALSE----------------------------------------
# 類別資料可視化R套件
library(vcd)

## ----eval=FALSE----------------------------------------------------------
## # 內建關節炎案例形式類別資料
## str(Arthritis)

## ------------------------------------------------------------------------
# 除了Age之外，其餘均為因子變數
head(Arthritis, 5)
# 觀測值總數
nrow(Arthritis)
# 變數個數
ncol(Arthritis)

## ------------------------------------------------------------------------
# 頻次形式類別資料
# 以expand.grid()建立sex與party的所有可能組合及其頻次
GSS <- data.frame(expand.grid(sex = c("female", "male"),
party = c("dem", "indep", "rep")), count = c(279, 165, 73,
47, 225, 191))
# 所有因子水準組合下的頻次形式
GSS
# 因子變數與頻次構成的表格
str(GSS)
# 總觀測值個數
sum(GSS$count)
# 各因子所有水準的組合數
nrow(GSS)

## ------------------------------------------------------------------------
# 內建的鐵達尼號四維列聯表
str(Titanic) # class 'table'
# 螢幕與紙張為平面，此處用4張(Why?)二維列聯表呈現
Titanic
# 觀測值總數
sum(Titanic)
# 四個因子變數下的水準
dimnames(Titanic)
# 表格維度
length(dimnames(Titanic))
# 各因子變數(各維度)的水準數(各維長度)，i.e. 表格大小
sapply(dimnames(Titanic), length)
# 轉成長表，即頻次形式
df <- as.data.frame(Titanic)
head(df)

## ----eval=FALSE----------------------------------------------------------
## #請與前面GSS資料物件做比較
## str(df)

## ------------------------------------------------------------------------
# 也是觀測值總數
sum(df$Freq)

## 3.5.3 類別變數視覺化關聯檢驗 ------------------------------------------------------------------------
# 注意類別為'table'的表格形式
str(HairEyeColor)
# 兩張二維(4*4)列聯表
HairEyeColor
# 總觀測數
sum(HairEyeColor)
# dimnames()傳回何種資料結構？
length(dimnames(HairEyeColor))
# 各維長度
sapply(dimnames(HairEyeColor), length)

## ----warning=FALSE, message=FALSE----------------------------------------
# 髮色、眼色與性別三維列聯表HairEyeColor
data(HairEyeColor)
# 沿著Sex加總Hair-Eye邊際頻次，並變更為Eye-Hair的呈現方式
(tab <- margin.table(HairEyeColor, c(2,1)))
# 載入類別資料視覺化套件，為了呼叫sieve()
library(vcd)

## ----fig.align="center", fig.cap = "\\label{fig:sieve_diagram}眼色-髮色濾網圖"----
# 表格傳入繪製濾網圖
sieve(tab, shade = TRUE)

## ------------------------------------------------------------------------
# 加州大學柏克萊分校入學不公資料集
data(UCBAdmissions)
# 六張二維(2*2)列聯表
UCBAdmissions

## ----eval=FALSE----------------------------------------------------------
## # 前述表格形式
## str(UCBAdmissions)

## ------------------------------------------------------------------------
# 表格重排為Gender-Admit-Dept
x <- aperm(UCBAdmissions, c(2, 1, 3))
x

## ----eval=FALSE----------------------------------------------------------
## # 因子變數順序變動
## str(x)

## ------------------------------------------------------------------------
# 報章雜誌常見的高維列聯表呈現方式(flatten table)
ftable(x)
# 依Dept(第3維)加總邊際和
margin.table(x, c(1, 2))

## ----fig.align="center", fig.cap = "\\label{fig:fourfold}不分系所之性別 versus 入學與否的四重圖"----
# 不分系所的Gender-Admit四重圖
fourfold(margin.table(x, c(1, 2)))

## ----fig.align="center", fig.cap = "\\label{fig:stratified_fourfolds}各系所之性別 versus 入學與否的四重圖"----
# 以系所為層別的Gender-Admit四重圖
fourfold(x)

