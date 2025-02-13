## Collated by Ching-Shih Tsou 鄒慶士 博士 (Ph.D.) Distinguished Prof. at the Department of Mechanical Engineering/Director at the Center of Artificial Intelligence & Data Science (機械工程系特聘教授兼人工智慧暨資料科學研究中心主任), MCUT (明志科技大學); Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), NTUB (國立臺北商業大學)
## Notes: This code is provided without warranty.

## 6.2.2 多層感知機 ####
## 6.2.2.1 混凝土強度估計案例 ------------------------------------------------------------------------
# R語言讀入混凝土配方與強度資料
concrete <- read.csv("./_data/concrete.csv")
# 水泥、爐渣、煤灰、水、超塑料、粗石、細砂、歷時與強度等變數
str(concrete)
# 各變數量綱有差距
summary(concrete)
# 成批繪製直方圖，有些變數非對稱有偏斜的狀況，目標變數近似常態
op <- par(mfrow = c(3,3))
lapply(names(concrete), function(u) {hist(concrete[[u]], main = paste("Histogram of ", names(concrete[u])), xlab = "", cex.main = 0.8)})
par(op)

## ------------------------------------------------------------------------
# 定義0-1正規化函數normalize
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}
# 運用隱式迴圈函數逐欄(含y)正規化後再將等長串列轉為資料框
concrete_norm <- as.data.frame(lapply(concrete, normalize))

## ------------------------------------------------------------------------
# 轉換前後的摘要統計值
summary(concrete$strength)
summary(concrete_norm$strength)

## ------------------------------------------------------------------------
# 訓練與測試資料切分(確認樣本順序已為隨機)
concrete_train <- concrete_norm[1:750, ]
concrete_test <- concrete_norm[751:1030, ]

# summary(concrete_norm$strength)
# summary(concrete_train$strength)
# summary(concrete_test$strength)

## ----warning=FALSE, message=FALSE----------------------------------------
# 載入R語言類神經網路簡易套件{neuralnet}
library(neuralnet)
concrete_model <- neuralnet(formula = strength ~ cement +
slag + ash + water + superplastic + coarseagg + fineagg +
age, data = concrete_train, hidden = c(1))

concrete_model <- neuralnet(formula = strength ~ cement +
slag + ash + water + superplastic + coarseagg + fineagg +
age, data = concrete_train, hidden = c(3,3))

## ----fig.align="center", fig.cap = "\\label{fig:concrete_single_hidden_neuron}單一隱藏層單一隱藏神經元類神經網路圖"----
# 網路拓樸視覺化
plot(concrete_model, rep = "best")

# 報表解讀與關係詮釋(https://stats.stackexchange.com/questions/280536/what-is-the-intuitive-meaning-of-generalized-weight-in-a-neural-net-and-how-does)
# The partial derivative of the log-odds function with respect to covariate of interest is the generalized weights for that covariate.
# A large variance of generalized weights for a covariate indicates non-linearity of its independent effect. If generalized weights of a covariate are approximately zero, the covariate is considered to have no effect on outcome."
options(scipen=999)
glwts <- data.frame(predictor = names(concrete)[1:8], generalized.weights = apply(concrete_model$generalized.weights[[1]], 2, var)) # generalized.weights:	a list containing the generalized weights of the neural network for every repetition.
glwts[order(glwts$generalized.weights),]

## ------------------------------------------------------------------------
# 以測試集評估模型績效
model_results <- compute(concrete_model, concrete_test[1:8])
predicted_strength <- model_results$net.result
# 預測集前六筆強度預測值
head(predicted_strength)

## ------------------------------------------------------------------------
cor(predicted_strength, concrete_test$strength)

# 請計算MSE
sum((predicted_strength - concrete_test$strength)^2)
sqrt(sum((predicted_strength - concrete_test$strength)^2))

## ---- fig.align='center', fig.cap = "\\label{fig:concrete_single_scatter}單層單隱藏神經元下，混凝土強度實際值對預測值的散佈圖"----
# 強度預測值與實際值的最小最大值
(axisRange <- extendrange(c(concrete_test$strength,
predicted_strength)))
# 在預測值與實際值散佈範圍中繪製散佈圖
plot(concrete_test$strength ~ predicted_strength, ylim =
axisRange, xlim = axisRange, xlab = "Predicted", ylab =
"Observed")
# 加45度角斜直線
abline(0, 1, col = 'darkgrey', lty = 2)

## ------------------------------------------------------------------------
# hidden=5增加單一隱藏層內的神經元
concrete_model2 <- neuralnet(strength ~ cement + slag + ash +
water + superplastic + coarseagg + fineagg + age, data =
concrete_train, hidden = 5)

## ----fig.align="center", fig.cap = "\\label{fig:concrete_multiple_hidden_neurons}單一隱藏層多個隱藏神經元類神經網路圖"----
# 多個隱藏神經元的網路拓樸視覺化
plot(concrete_model2, rep = "best")
# 報表解讀與關係詮釋
glwts2 <- data.frame(predictor = names(concrete)[1:8], generalized.weights = apply(concrete_model2$generalized.weights[[1]], 2, var))
glwts2[order(glwts2$generalized.weights),]

## ------------------------------------------------------------------------
# 以測試集評估新模型績效
model_results2 <- compute(concrete_model2,concrete_test[1:8])
predicted_strength2 <- model_results2$net.result
# 預測值與實際值相關程度提高
cor(predicted_strength2, concrete_test$strength)

## ---- fig.align='center', fig.cap = "\\label{fig:concrete_multiple_scatter}單層多隱藏神經元下，混凝土強度實際值對預測值的散佈圖"----
# 實際值對預測值散佈圖
axisRange <- extendrange(c(concrete_test$strength,
predicted_strength2))
plot(concrete_test$strength ~ predicted_strength2, ylim =
axisRange, xlim = axisRange, xlab = "Predicted", ylab =
"Observed")
abline(0, 1, col = 'darkgrey', lty = 2)


## 6.3 強化式學習 ####
## ----warning=FALSE, message=FALSE----------------------------------------
# 載入R語言馬可夫決策過程套件
library(MDPtoolbox)
# 上行行動的狀態轉移機率矩陣
(up = matrix(c( 1, 0, 0, 0,
             0.7, 0.2, 0.1, 0,
             0, 0.1, 0.2, 0.7,
             0, 0, 0, 1),
          nrow = 4, ncol = 4, byrow = TRUE))
# 下行行動的狀態轉移機率矩陣
(down = matrix(c(0.3, 0.7, 0, 0,
              0, 0.9, 0.1, 0,
              0, 0.1, 0.9, 0,
              0, 0, 0.7, 0.3),
            nrow = 4, ncol = 4, byrow = TRUE))
# 左行行動的狀態轉移機率矩陣
(left = matrix(c( 0.9, 0.1, 0, 0,
               0.1, 0.9, 0, 0,
               0, 0.7, 0.2, 0.1,
               0, 0, 0.1, 0.9),
            nrow = 4, ncol = 4, byrow = TRUE))
# 右行行動的狀態轉移機率矩陣
(right = matrix(c( 0.9, 0.1, 0, 0,
                0.1, 0.2, 0.7, 0,
                0, 0, 0.9, 0.1,
                0, 0, 0.1, 0.9),
             nrow = 4, ncol = 4, byrow = TRUE))
# 結合為行動集矩陣
Actions = list(up=up, down=down, left=left, right=right)

## ------------------------------------------------------------------------
# 定義報酬矩陣(負值表懲罰，正值是獎勵)
(Rewards = matrix(c( -1, -1, -1, -1,
                  -1, -1, -1, -1,
                  -1, -1, -1, -1,
                  10, 10, 10, 10),
               nrow = 4, ncol = 4, byrow = TRUE))
# 迭代式評估各種可能的政策
solver = mdp_policy_iteration(P = Actions, R = Rewards,
discount = 0.1)

## ------------------------------------------------------------------------
# 最佳政策：2(下)4(右)1(上)1(上)
solver$policy
names(Actions)[solver$policy]
# 迭代次數
solver$iter
# 求解時間
solver$time

