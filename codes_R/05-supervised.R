## Collated by Ching-Shih Tsou 鄒慶士 博士 (Ph.D.) Distinguished Prof. at the Department of Mechanical Engineering/Director at the Center of Artificial Intelligence & Data Science (機械工程系特聘教授兼人工智慧暨資料科學研究中心主任), MCUT (明志科技大學); Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), NTUB (國立臺北商業大學)
## Notes: This code is provided without warranty.

## 5.1 線性迴歸與分類 ####
## 5.1.1 多元線性迴歸 ------------------------------------------------------------------------
library(AppliedPredictiveModeling)
# 載入溶解度資料的數個資料物件
data(solubility)
# 資料物件名都是以solT開頭的名稱
ls(pattern = "^solT")

## ------------------------------------------------------------------------
# 計算樣本總數
nrow(solTrainXtrans) + nrow(solTestXtrans)
# 預測變數個數
ncol(solTrainXtrans)

## ------------------------------------------------------------------------
# 合併屬性矩陣X與類別標籤向量y，產生統計人群慣用的資料表
trainingData <- solTrainXtrans
trainingData$Solubility <- solTrainY
# R語言線性迴歸建模的主要函數lm()
lmFitAllPredictors <- lm(Solubility ~ ., data = trainingData)
# 配適好的模型其類別與建模函數同名
class(lmFitAllPredictors)
# summary.lm()產生R語言線性迴歸摘要報表lmAllRpt
lmAllRpt <- summary(lmFitAllPredictors)
# 229個迴歸係數報表很長，僅挑出其t檢定顯著水準低於5%者
sigVars <- lmAllRpt$coefficients[,"Pr(>|t|)"] < .05
# 54個係數的顯著水準低於5%(至少一星*)
sum(sigVars)
# 邏輯值索引只檢視54個顯著的迴歸係數
# 更新原229項龐大的迴歸係數報表
lmAllRpt$coefficients <- lmAllRpt$coefficients[sigVars, ]
lmAllRpt

## ------------------------------------------------------------------------
# predict.lm()預測測試樣本溶解度
lmPred1 <- predict(lmFitAllPredictors, solTestXtrans)
head(lmPred1)

## ------------------------------------------------------------------------
# 測試集實際值與預測值組成資料框
lmValues1 <- data.frame(obs = solTestY, pred = lmPred1)
library(caret)
# R語言{caret}套件績效評量計算函數
defaultSummary(lmValues1)
# library(psych)
# describe(solTestY)
# vars   n mean   sd median trimmed  mad    min  max range
# X1    1 316 -2.8 2.08  -2.48   -2.64 1.92 -10.41 1.07 11.48
# skew kurtosis   se
# X1 -0.73     0.47 0.12

## ------------------------------------------------------------------------
# 後向式逐步迴歸step()需傳入完整模型，再逐次剔除不重要變數
# 建模時間長，system.time()衡量程式碼執行時間(1.9節)
# 逐步迴歸建模過程AIC或BIC值愈小，模型配適的愈好(3.2.1節)
system.time(reducedSolMdl <- step(lmFitAllPredictors,
direction = 'backward'))
# drop1(lmFitAllPredictors) # step()跑了很多次的drop1() !
# 儲存執行耗時的配適結果
# save(reducedSolMdl, file = "reducedSolMdl.RData")
# 因模型建立耗時，載入預先跑好的模型物件
load("./_data/reducedSolMdl.RData")
# 原始報表過長，請讀者自行執行程式碼
# summary(reducedSolMdl)

## ----eval=FALSE----------------------------------------------------------
## # 129個模型項代表原228個變數，後向式逐步迴歸挑選了128個入模
## str(coef(reducedSolMdl))

## ------------------------------------------------------------------------
# 後向式逐步迴歸摘要報表lmBackRpt
lmBackRpt <- summary(reducedSolMdl)
# 因迴歸報表很長，挑出t檢定顯著水準低於5%的模型項
sigVars <- lmBackRpt$coefficients[,"Pr(>|t|)"] < .05
# 96個入模變數的係數顯著水準低於5%(至少一星*)
sum(sigVars)
# 更新原129項的後向式逐步迴歸係數報表
lmBackRpt$coefficients <- lmBackRpt$coefficients[sigVars, ]

## ----eval=FALSE----------------------------------------------------------
## # 更新後的迴歸報表仍然相當長
## lmBackRpt

## ------------------------------------------------------------------------
# 先建立只有截距項的最簡模型
minSolMdl <- lm(Solubility ~ 1, data = trainingData)
# 前向式逐步迴歸step()需傳入最簡模型，再逐次增加變數入模
# as.formula()設定scope引數的最複雜模型公式
system.time(fwdSolMdl <-step(minSolMdl, direction='forward'
, scope = as.formula(paste("~", paste(names(solTrainXtrans)
, collapse = "+"))), trace=0))
# 儲存執行耗時的配適結果
# save(fwdSolMdl, file = "fwdSolMdl.RData")
# 因模型建立耗時，載入預先跑好的模型物件
load("./_data/fwdSolMdl.RData")
# 原始報表過長，請讀者自行執行程式碼
# summary(fwdSolMdl)

## ----eval=FALSE----------------------------------------------------------
## # 115個模型項代表原228個變數，前向式逐步迴歸挑選了114個入模
## str(coef(fwdSolMdl))

## ------------------------------------------------------------------------
# 前向式逐步迴歸摘要報表lmFwdRpt
lmFwdRpt <- summary(fwdSolMdl)
# 因迴歸報表很長，以t檢定顯著水準低於5%的標準縮減報表
sigVars <- lmFwdRpt$coefficients[,"Pr(>|t|)"] < .05
# 76個入模變數的係數顯著水準低於5%(至少一星*)
sum(sigVars)
# 更新原115項的前向式逐步迴歸係數報表
lmFwdRpt$coefficients <- lmFwdRpt$coefficients[sigVars, ]

## ----eval=FALSE----------------------------------------------------------
# 更新後的報表仍然相當長
lmFwdRpt

## ----eval=FALSE----------------------------------------------------------
# 前向式與後向式(較大)逐步迴歸模型ANOVA比較(結果顯著)
# ANOVA報表中1: fwdSolMdl, 2: reducedSolMdl
anova(fwdSolMdl, reducedSolMdl)
# 請自行檢視實際呼叫之檢定函數的說明文件
# ?anova.lm()

## ----echo=FALSE----------------------------------------------------------
data.frame(anova(fwdSolMdl, reducedSolMdl), check.names = FALSE)

## ----eval=FALSE----------------------------------------------------------
# 後向式與完整(較大)逐步迴歸模型ANOVA比較(結果不顯著)
# ANOVA報表中1: reducedSolMdl, 2: lmFitAllPredictors
anova(reducedSolMdl, lmFitAllPredictors)

## ----echo=FALSE----------------------------------------------------------
data.frame(anova(reducedSolMdl, lmFitAllPredictors), check.names = FALSE)

## 5.1.2 偏最小平方法迴歸 ####
## ----warning=FALSE, message=FALSE----------------------------------------
# 載入R語言偏最小平方法估計套件
library(pls)
# 模型公式語法擬合模型
plsFit <- plsr(Solubility ~ ., data = trainingData)
# 凡事總有例外，mvr類別物件
class(plsFit)
# 擬合結果摘要報表
summary(plsFit)

## ----fig.align="center", fig.cap = "\\label{fig:pls_screeplot}RMSEP對成份個數的陡坡圖"----
# 繪製PLS決定主成份個數的陡坡圖
# plottype引數決定繪製不同主成份下的核驗統計值(預設為RMSEP)
plot(plsFit, plottype = "validation")

## ----fig.align="center", fig.cap = "\\label{fig:pls_screeplot}PLS在9個主成份下預測值對實際值的散佈圖"----
# 繪製9個PLS主成份下，訓練集之預測值對實際值的散佈圖
plot(plsFit, ncomp = 9)
# 9個PLS主成份下，訓練集之預測值對實際值的相關係數
cor(plot(plsFit, ncomp = 9)[,"measured"], plot(plsFit,
ncomp = 9)[,"predicted"]) # Try ncomp = 10 !
# 9個PLS主成份下的模型，預測測試集樣本solTestXtrans溶解度
pre <- predict(plsFit, solTestXtrans, ncomp = 9)
# 測試集預測值與實際值的相關係數
cor(pre[, 1, 1], solTestY) # 請自行檢視str(pre)

## 5.1.3 脊迴歸、LASSO迴歸與彈性網罩懲罰模型 ------------------------------------------------------------------------
# 設定待調懲罰係數值
ridgeGrid <- expand.grid(lambda = seq(0, .1, length = 15))
# 十摺交叉驗證參數調校訓練與測試
ctrl <- trainControl(method = "cv", number = 10)
set.seed(100)
# Windows作業系統多核運算(https://waterprogramming.wordpress.com/2020/03/16/parallel-processing-with-r-on-windows/)
# library(doParallel) # for Windows
# library(parallel) # library(snow) # for Windows
# no_cores <- detectCores(logical = TRUE)
# workers <- makeCluster(4) # for Windows (remove  type="sock")
# registerDoParallel(workers) # registerParallel(workers) # for Windows
# MacOS或Linux作業系統多核運算
# library(doMC) # for Mac & Linux
# registerDoMC(cores = 8) # for Mac & Linux
# 請注意本節train()函數並未使用模型公式語法，與第三章不同
system.time(ridgeTune <- train(
x = solTrainXtrans, # 校驗集屬性矩陣
y = solTrainY, # 類別標籤向量
method = "ridge", # 訓練方法
tuneGrid = ridgeGrid, # 待調參數網格
trControl = ctrl, # 訓練測試機制
preProc=c("center","scale"))) # 前處理方式
# 儲存耗時參數校驗結果
# save(ridgeTune, file = "ridgeTune.RData")

## ----fig.align="center", fig.cap = "\\label{fig:ridge_tuningplot}脊迴歸不同懲罰係數下的模型績效概況圖"----
# 因模型建立與調校耗時，載入預先跑好的模型物件
load("./_data/ridgeTune.RData")
ridgeTune
# 不同懲罰係數下，十摺交叉驗證平均RMSE折線圖
plot(ridgeTune, xlab = 'Penalty')

## ------------------------------------------------------------------------
# 兩個待調參數形成的3*20網格
enetGrid <- expand.grid(lambda = c(0, 0.01, .1), fraction
= seq(.05, 1, length = 20))
set.seed(100)
# 引數訓練方法method改為enet
# system.time(enetTune <- train(x = solTrainXtrans,
              # y = solTrainY,
              # method = "enet",
              # tuneGrid = enetGrid,
              # trControl = ctrl,
              # preProc = c("center", "scale")))
# 儲存耗時參數校驗結果
# save(enetTune, file = "enetTune.RData")

## ----fig.align="center", fig.cap = "\\label{fig:enets_tuningplot}彈性網罩模型不同參數組合下的模型績效概況圖"----
# 因模型建立與調校耗時，載入預先跑好的模型物件
load("./_data/enetTune.RData")
enetTune
# 參數調校模型物件的類別為train
class(enetTune)
# 不同參數組合下，交叉驗證績效概況
plot(enetTune)

## 5.1.4 線性判別分析 ------------------------------------------------------------------------
# 第1類樣本母體平均值向量
(mu1 <- c(1,1))
# 第2類樣本母體平均值向量
(mu2 <- c(3.5,2))
# 兩類樣本共同的母體共變異數矩陣
(sig <- matrix(c(1,0.85,0.85,2), ncol = 2))

## ------------------------------------------------------------------------
# 載入R語言多元常態分佈隨機抽樣函數
library(mvtnorm)
n1 <- 1000*0.9
n2 <- 1000*0.1
# 定義0-1類別標籤向量，0與1各重複n1與n2次
group <- c(rep(0,n1), rep(1,n2))
set.seed(130)
# 模擬抽出第1類的二維訓練樣本
X1train <- rmvnorm(n1, mu1, sig)
# 模擬抽出第2類的二維訓練樣本
X2train <- rmvnorm(n2, mu2, sig)
# 合併兩類訓練樣本為屬性矩陣
Xtrain <- rbind(X1train, X2train)
# 屬性矩陣與類別標籤組織為訓練資料集
dtrain <- data.frame(X = Xtrain, group = group)
dtrain[898:903,]
set.seed(131)
# 測試資料集同前模擬與處理
X1test <- rmvnorm(n1,mu1,sig)
X2test <- rmvnorm(n2,mu2,sig)
Xtest <- rbind(X1test,X2test)
dtest <- data.frame(X = Xtest,group = group)
dtest[898:903,]

## ----fig.align="center", fig.cap = "\\label{fig:rmvnorm_scatterplot}訓練與測試兩類樣本的散佈圖"----
# 觀察兩類樣本在各子集散佈狀況是否相似
op <- par(mfrow = c(1,2))
plot(dtrain$X.1, dtrain$X.2, pch = dtrain$group + 1, main =
"Training data", xlab = expression(x[1]),
ylab = expression(x[2]))
legend("bottomright", c("Cl1","Cl2"), pch = 1:2)
plot(dtest$X.1, dtest$X.2, pch = dtest$group + 1, main =
"Test data",xlab = expression(x[1]),ylab = expression(x[2]))
legend("bottomright", c("Cl1","Cl2"), pch = 1:2)
par(op)

## ------------------------------------------------------------------------
# 擬取用套件{MASS}下的lda()
library(MASS)
# 訓練模型
resLDA <- lda(group~., data = dtrain)
# 預測測試資料的類別隸屬度
predLDA <- predict(resLDA, newdata = dtest)$class
# 混淆矩陣
table(dtest$group, predLDA)
# 正確率計算
mean(dtest$group == predLDA)

## 5.1.5 羅吉斯迴歸分類與廣義線性模型 ####
## ----warning=FALSE, message=FALSE----------------------------------------
# 誤差分佈為二項式時，連結函數預設為logit
resLR <- glm(group~., data = dtrain, family = binomial)
# 羅吉斯迴歸的預測值是關心事件發生機率的logit值
predLogit <- predict(resLR, newdata = dtest)
head(predLogit)
# 套件{boot}的inv.logit()函數，將預測值逆轉換回機率值
library(boot)
predProb <- inv.logit(predLogit)
head(predProb)
# 以0.5為門檻值，機率再轉成類別標籤預測值
predLabel <- predProb > 0.5
head(predLabel)
# 混淆矩陣
table(dtest$group, predLabel)
# 正確率計算
mean(dtest$group == predLabel)


## 5.2.4 分類與迴歸樹 ####
## ----warning=FALSE, message=FALSE----------------------------------------
# R語言Recursive PARTitioning遞迴分割建樹套件
library(rpart)
head(iris)
# 以鳶尾花花瓣花萼長寬預測花種
iristree <- rpart(Species ~ ., data = iris)

## ----eval=FALSE----------------------------------------------------------
# 分類樹模型報表，與樹狀圖的說明相同
iristree

## ----fig.align="center", fig.cap = "\\label{fig:iris_ct}鳶尾花資料集分類樹"----
# 載入R語言樹狀模型繪圖套件{rpart.plot}，輕鬆視覺化分類樹模型
library(rpart.plot)
rpart.plot(iristree, digits = 3)

## ------------------------------------------------------------------------
# 匯入報稅稽核資料集，納稅義務人人口統計變數與稽核結果
audit <- read.csv('./_data/audit.csv')
head(audit)
# 目標變數稽核後是否有修正的(0: 無，1: 有)次數分佈
table(audit$TARGET_Adjusted)

## ------------------------------------------------------------------------
# 載入模型物件ct.audit
load("./_data/ct.audit.RData")

## ----eval=FALSE----------------------------------------------------------
# 分類樹模型報表(Too wide to show here. 參見圖5.17)
ct.audit

## ----message=FALSE, warning=FALSE, fig.align="center", fig.cap = "\\label{fig:before_pruned_ct}修剪前的分類樹"----
# 修剪前分類樹模型視覺化
library(rpart.plot)
rpart.plot(ct.audit, roundint = FALSE)

## ----CP------------------------------------------------------------------
# 取出複雜度參數表(表5.5 cptable)
knitr::kable(
  ct.audit$cptable, caption = '分類樹複雜度參數表',
  booktabs = TRUE
)

## ------------------------------------------------------------------------
# 從cptable定位交叉驗證錯誤率的最小值
(opt <- which.min(ct.audit$cptable[,"xerror"]))
# 從cptable定位相對錯誤率小於交叉驗證的最小錯誤率
# ，加上其對應的一倍標準誤
(oneSe <- which(ct.audit$cptable[, "rel error"] <
ct.audit$cptable[opt, "xerror"] +
ct.audit$cptable[opt, "xstd"])[1])
# 取得one-SE準則下的最佳CP值
(cpOneSe <- ct.audit$cptable[oneSe, "CP"])
# cpOneSe輸入prune()函數，完成one-SE事後修剪
ct.audit_pruneOneSe <- prune(ct.audit, cp = cpOneSe)

## ----fig.align="center", fig.cap = "\\label{fig:after_pruned_ct}修剪後的分類樹"----
# 繪製修剪後的分類樹圖形
rpart.plot(ct.audit_pruneOneSe, roundint = FALSE)


