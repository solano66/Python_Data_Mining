## Collated by Ching-Shih Tsou 鄒慶士 博士 (Ph.D.) Distinguished Prof. at the Department of Mechanical Engineering/Director at the Center of Artificial Intelligence & Data Science (機械工程系特聘教授兼人工智慧暨資料科學研究中心主任), MCUT (明志科技大學); Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), NTUB (國立臺北商業大學)
## Notes: This code is provided without warranty.

## 1.1 套件管理 -----------------------------------------------------------------
# 已載入記憶體之R套件
search()

## ------------------------------------------------------------------------
# 已安裝套件報表又寬又長，只顯示前六筆(head())結果的部分內容
head(installed.packages()[,-c(2, 5:8)])

## ----eval=FALSE----------------------------------------------------------
## # str()檢視install.packages()傳回的結果物件結構
## # 各套件16項資訊組成的字串矩陣
str(installed.packages())

## ----eval=FALSE----------------------------------------------------------
## # 套件存放路徑
## .libPaths()
## # [1] "/Library/Frameworks/R.framework/Versions/3.5/Resources
## # /library"

## 1.1.1 基本套件 ------------------------------------------------------------------------
# 有看到{stats}(R核心開發團隊編寫和維護)
search()
# 美國各州暴力犯罪率資料集前六筆數據(屬於基本套件{datasets})
head(USArrests)
# 標準化各變數向量
USArrests_z <- scale(USArrests)
# 逕行使用{stats}下的hclust()
# dist()函數計算兩兩州之間的歐幾里德直線距離
# 依州間距離方陣，對各州進行階層式集群(參見4.3.2節)
hc <- hclust(dist(USArrests_z), method = "average")

## ----fig.align="center", fig.cap = "\\label{fig:USArrests_dendrogram}美國50州犯罪與人口數據階層式集群樹狀圖"----
plot(hc, hang = -1, cex = 0.8)
# 所使用的scale(), dist(), hclust()均屬於基本套件{base}和{stats}

## 1.1.2 建議套件 ------------------------------------------------------------------------
# 沒有看到{lattice}
search()
# grep()函數應該在search()的結果中抓不到lattice
# 輸出character(0)表示結果沒有套件{lattice}
grep("lattice", search(), value=TRUE)

## ------------------------------------------------------------------------
# 檢視已下載套件清單
head(rownames(installed.packages()))
# 查核硬碟中是否有"lattice"，value=TRUE表示傳回匹配到的元素值
grep("lattice", rownames(installed.packages()), value = TRUE)

## ------------------------------------------------------------------------
# 載入建議套件
library(lattice)

## ----results='hide'------------------------------------------------------
# 資料集barley結構，4個變數除了yield其餘都是類別(因子)變數
str(barley)

## ------------------------------------------------------------------------
# barley前六筆數據
head(barley)

## ----fig.align="center", fig.cap = "\\label{fig:barley_dotplot}不同地區各年十種麥種的產量點圖", fig.height=11, fig.width=10----
# 克里夫蘭點圖繪製，多維列聯表視覺化繪圖方法
dotplot(variety ~ yield | year * site, data = barley)

## 1.1.3 貢獻套件 ------------------------------------------------------------------------
# data()載入資料集出現警告訊息，因為RAM中沒有其所依附的套件
# data(team.batting.00to08)
# Warning message: In data(team.batting.00to08) : data set
# 'team.batting.00to08' not found
# str()檢視資料結構時出現錯誤，因為RAM中根本沒有該資料集
# str(team.batting.00to08)
# Error in str(team.batting.00to08) : object
# 'team.batting.00to08' not found

## ------------------------------------------------------------------------
# 記憶體載入套件錯誤，因為硬碟中沒有該貢獻套件
# library(nutshell) # Error in library(nutshell) :
# there is no package called ‘nutshell’

## ------------------------------------------------------------------------
# 未載入{nutshell}到記憶體
search()

## ----eval=FALSE----------------------------------------------------------
## # 傳回character(0)，表示未下載{nutshell}到硬碟
## grep("nutshell", rownames(installed.packages()), value = TRUE)

## ----warning=FALSE, message=FALSE----------------------------------------
# 一部曲套件下載也可以透過RStudio GUI完成
# install.packages("nutshell")
# 二部曲套件載入
library(nutshell)
# 取用套件中資料集
data(team.batting.00to08)

## ----results='hide'------------------------------------------------------
# 可以檢視資料結構了
str(team.batting.00to08)

## 1.2 環境與輔助說明 ------------------------------------------------------------------------
# 在當前的環境中將符號"x"與物件168關聯起來
(x <- 168)
# 同一環境中將符號"y"與物件2關聯起來
(y <- 2)
# 符號"x"與"y"再組成"z"
(z <- x*y)

## ------------------------------------------------------------------------
# 都是套件，沒有資料集
search()
# 基本套件{datasets}中1947到1962年7個經濟變數資料集
longley$GNP # longley未附加在搜尋路徑前的引用語法
# 在全域環境中附加上資料集longley
attach(longley)
# 有看到資料集longley
search()
# 資料集附加在搜尋路徑後的引用語法(無需加上資料集名稱！)
GNP
# 全域環境中卸載資料集longley
detach(longley)
# 卸載後沒見到資料集longley
search()
# 也可以卸載套件
detach(package:nutshell)
# 卸載後沒見到套件{nutshell}
search()

## ------------------------------------------------------------------------
# 查詢全域環境中的物件
objects()
ls()
# 查詢基本環境中的物件，只顯示後六筆
tail(objects(envir = baseenv()))

## ----results='hide'------------------------------------------------------
# 以getwd()查詢當前工作目錄，並儲存為iPAS字串物件
(iPAS <- getwd())

## ------------------------------------------------------------------------
# 設定工作目錄為MAC OS下的家目錄
setwd("~")
# 取得我的家目錄
getwd()
# 還原iPAS目錄
setwd(iPAS)

## ----results='hide'------------------------------------------------------
# 確定工作目錄已變更
getwd()
# [1] "/Users/Vince/cstsouMac/BookWriting/bookdown-chinese-
# master"

## ------------------------------------------------------------------------
# R語言網頁版線上使用說明
# help.start()

## ------------------------------------------------------------------------
# help(plot)
# ?plot

## ------------------------------------------------------------------------
# help.search("plot")
# ??plot # an alias of help.search

## ------------------------------------------------------------------------
# 名稱中具有字串"plot"的函數
head(apropos("plot"), 10)
# 名稱中具有字串"plot"的套件
find("plot")
# 擴大搜尋名稱中具有字串"plot"的套件
find("plot", simple.words = FALSE)

## 1.3 R語言資料物件 ####
## 1.3.1 向量 #### 
## ----warning=FALSE, message=FALSE----------------------------------------
# 套件{UsingR}內含資料集bumpers, firstchi與crime
library(UsingR)
firstchi

## ------------------------------------------------------------------------
# 查閱資料集使用說明，後續不再列出
# help(firstchi)
# 因為是實數值向量，所以類別型態是numeric
class(firstchi)
# NULL表各元素無名稱
names(firstchi)

## ------------------------------------------------------------------------
# 留意具名向量的呈現方式，並無各列最左元素的編號
bumpers

# 類別型態同樣是numeric
class(bumpers)
# 具名向量元素名稱
names(bumpers)

## ------------------------------------------------------------------------
# 以R語言冒號運算子':'創建向量
(a <- 0L:4L)
# 也可以用向量創建函數c()
(a <- c(0,1,2,3,4))
# 因為是整數值向量，所以類別型態是integer
class(a)

## ------------------------------------------------------------------------
# 創建字串向量
(b <- c("one", "two", "three", "four", "five"))
# 因為是字串向量，所以類別型態是character
class(b)
# 創建邏輯值向量
(c <- c(TRUE, TRUE, TRUE, F, T, F))
# 因為是邏輯值向量，所以類別型態是logical
class(c)

## ------------------------------------------------------------------------
# 數值強制轉換為字串
(d <- c("one", "two", "three", 4, 5))
# 型別強制轉換後類別型態是character
class(d)
# 邏輯值強制轉換為數值
(e <- c(1, 0, TRUE, FALSE))
# 型別強制轉換後類別型態是numeric
class(e)
# 邏輯值強制轉換為字串
(f <- c("one", "zero", TRUE, FALSE))
# 型別強制轉換後類別型態是character
class(f)

## 1.3.2 矩陣 ------------------------------------------------------------------------
# 美國50州統計數據
head(state.x77)
# 因為是矩陣，所以類別型態是matrix
class(state.x77)
# 查詢維度名稱
dimnames(state.x77)
# 注意!矩陣無元素或變數名稱
names(state.x77)

## ------------------------------------------------------------------------
# R語言內部觀點
typeof(state.x77)
# S語言的觀點
mode(state.x77)
# 與R物件編譯任務有關
storage.mode(state.x77)

## ------------------------------------------------------------------------
# 橫列名向量長度5
rnames <- paste0("row", 1:5)
# 縱行名向量長度4
cnames <- paste0("col", 1:4)
# 注意橫列名與縱行名(even兩者長度相同也是)組成串列
(y <- matrix(1:20, nrow = 5, ncol = 4, dimnames =
list(rnames, cnames)))
# 注意無行列名稱之矩陣呈現方式([列編號,]與[,行編號])
(y <- matrix(1:20, nrow = 5, ncol = 4))

## 1.3.3 陣列 ------------------------------------------------------------------------
# 四張4乘2的二維表格(四維物件如何在二維平面和三維空間中呈現呢？)
Titanic
# 高維陣列物件的類別為table
class(Titanic)
# 各維因子變數水準名(參見1.3.6節因子)
dimnames(Titanic)
# 扁平式四維列聯表，與前面的擺放方式不同而已
ftable(Titanic)

## ------------------------------------------------------------------------
# 各維(因子水準)名稱向量
dim1 <- c("A1", "A2")
dim2 <- c("B1", "B2", "B3")
dim3 <- c("C1", "C2", "C3", "C4")
# 四個2乘3二維矩陣
# 請思考dim和dimnames哪個是向量？哪個是串列？Why?)
(z <- array(1:24, dim = c(2, 3, 4), dimnames =
list(dim1, dim2, dim3)))

## 1.3.4 串列 ------------------------------------------------------------------------
# 三個元素的串列
Harman23.cor

## ----results="hide"------------------------------------------------------
# 留意cov元素下方有矩陣維度名稱屬性"dimnames"
str(Harman23.cor)

## ------------------------------------------------------------------------
# 以names()函數取出串列元素名稱
names(Harman23.cor)

## ------------------------------------------------------------------------
# 五花八門的串列元素g,h,j,k
g <- "My First List"
h <- c(25, 26, 18, 39)
j <- matrix(1:10, nrow = 5, byrow = T)
k <- c("one", "two", "three")
# 注意有給定和未給定元素名稱的語法差異與顯示差異
(mylist <- list(title = g, ages = h, j, k))

## 1.3.5 資料框 ------------------------------------------------------------------------
# 資料框外表看似矩陣
head(crime)
# 返回類別值既非matrix亦非list，但須注意與這兩類物件的異同
class(crime)

## ------------------------------------------------------------------------
# 串列物件的各種類別值返回函數之結果均相同
typeof(crime)
mode(crime)
storage.mode(crime)
# 資料框實際上以串列的方式儲存各欄等長的向量
as.list(crime) # 打回原形！

## ------------------------------------------------------------------------
# 視為串列，傳回變數名稱
names(crime) # 想想上面as.list(crime)的結果
# 視為矩陣，傳回二維維度名稱
dimnames(crime)

## ------------------------------------------------------------------------
# 將crime資料框強制轉為矩陣
crime_mtx <- as.matrix(crime)
# 顯示結果與資料框儲存方式一模一樣！
head(crime_mtx)
class(crime_mtx)

## ----results="hide"------------------------------------------------------
head(Cars93, n=3L)

## ----results='hide'------------------------------------------------------
# 檢視資料框結構，注意$開頭之各欄位的型別
str(Cars93)

## ------------------------------------------------------------------------
# 將Cars93資料框強制轉為矩陣
head(as.matrix(Cars93), 2)

## ------------------------------------------------------------------------
# 建立各欄位向量
patientID <- c(1, 2, 3, 4)
age <- c(25, 34, 28, 52)
diabetes <- c("Type1", "Type2", "Type1", "Type1")
status <- c("Poor", "Improved", "Excellent", "Poor")
# 注意省略欄位名稱時，自動產生欄名的方式
(patientdata <- data.frame(patientID, age, diabetes, status))

## ----results='hide'------------------------------------------------------
# 字串預設會轉為因子向量，注意diabetes與status
str(patientdata)

## ------------------------------------------------------------------------
# 改變預設設定為stringsAsFactors = F，注意前述字串欄位的型別
str(data.frame(patientID, age, diabetes, status,
stringsAsFactors = F))

## 1.3.6 因子 ------------------------------------------------------------------------
# 創建糖尿病類型字串向量
(diabetes <- c("Type1", "Type2", "Type1", "Type1"))
# 讀者請注意轉為因子類別後，與上方字串向量不同之處是少了雙引號，
# 及多了下方的詮釋資料(metadata) Levels: Type1 Type2
(diabetes <- factor(diabetes))
# 因子類別表面上看似類別，其實背後對應到數字了！
class(diabetes)
# as.numeric()可將因子向量打回原形，請思考何時會用到？
as.numeric(diabetes)
# 水準數(no. of levels)為2的次數分佈表，上方為水準
# (level) Type1與Type2，下方為次數(frequency)
table(diabetes)

## ------------------------------------------------------------------------
# 病患康復狀況status字串變數
(status <- c("Poor", "Improved", "Excellent", "Poor"))
# 設定有序因子的大小順序後轉為有序類別變數
# 注意有序因子與因子兩者的詮釋資料不同
(status <- factor(status, order = TRUE, levels = c("Poor",
"Improved", "Excellent")))
class(status)

## ----warning=FALSE, message=FALSE----------------------------------------
# 類別資料視覺化套件
library(vcd)
# 關節炎資料集
data(Arthritis)

## ----eval=FALSE----------------------------------------------------------
## # 編號、療法、性別、年齡、療癒狀況等變數
## str(Arthritis)

## ----warning=FALSE, message=FALSE----------------------------------------
# 單熱編碼R套件
library(onehot)
# 因為Treatment與Sex各有兩個水準，所以結果為四欄矩陣
(encoder <- onehot(Arthritis[c("Treatment", "Sex")]))
# 模型物件encoder類別值與建模函數名稱相同
class(encoder)
# 預測方法predict()根據模型encoder對兩類別欄位做編碼轉換
arthritisOh <- predict(encoder, Arthritis[c("Treatment",
"Sex")])
# 比對觀測值41到45編碼前後的結果(;分隔兩個指令)
Arthritis[41:45,c("Treatment", "Sex")]; arthritisOh[41:45,]
# 合併單熱編碼結果
arthritisOh <- cbind(Arthritis[c("ID", "Age", "Improved")],
arthritisOh)
head(arthritisOh)

## 1.3.7 R語言原生資料物件取值 ------------------------------------------------------------------------
# 冒號運算子建向量，注意列首元素左方編號
(x <- 20:16)
# 元素設定名稱
names(x) <- c("1st", "2nd", "3rd", "4th", "5th")
# 具名向量的呈現與不具名的不同
x
# 單一位置取值
x[4]
# R負索引值是去掉第四個，Python是倒數第四個！
# 參見圖1.8 Python語言前向與後向索引編號示意圖
x[-4]
# 單一名稱取值(如果x是具名向量)
x["4th"]
# 連續位置範圍取值
x[1:4]
# 連續位置範圍移除
x[-(1:4)]
# 位置間隔取值(注意位置的錯置)
x[c(1,4,2)]
# 位置重覆取值
x[c(1,2,2,3,3,3,4,4,4,4)]
# 多重名稱取值(如果x是具名向量)
x[c("1st","3rd")]

## ------------------------------------------------------------------------
# 邏輯值取值
x[c(T,T,F,F,F)]

## ------------------------------------------------------------------------
# 進階邏輯值取值(18重複了五次，接著就向量x中對應的元素比較)
x[x > 18]
# 邏輯陳述複合句
x[x > 16 & x < 19]
# 善用二元運算子%in%回傳的邏輯值
x[x %in% c(16, 18, 20)]

## ------------------------------------------------------------------------
# 1.3.4節的mylist
mylist
# 取出串列的第二個元素
mylist[[2]]
# 取出串列中名稱為ages的元素
mylist[["ages"]]
# 同樣可以取出串列中名稱為ages的元素
mylist$ages
# 取出串列第四個元素形成的子串列，注意結果帶有兩對中括弧
mylist[4]

## ------------------------------------------------------------------------
# 一對中括弧取出子串列物件
class(mylist[4])
# 兩對中括弧取出該元素類別的物件(此處為一維字串向量物件)
class(mylist[[4]])
# 取出串列第二到第三個元素形成的子串列，串列唯一可取多個元素的語法
mylist[2:3]
# 語法錯誤！不可以兩對中括弧取多個元素
# mylist[[1:2]]
# Error in mylist[[1:2]] : subscript out of bounds

## ------------------------------------------------------------------------
# matrix()函數創建二維矩陣
(x <- matrix(1:12, nrow = 3, ncol = 4))
# 行列命名
dimnames(x) <- list(paste("row", 1:3, sep = ''),
paste("col", 1:4, sep = ''))
# 注意具名矩陣(named matrix)呈現方式
x
# 取行列交叉下單一元素
x[3, 4]
# 取單列
x[3,]
# 類別是一維向量物件
class(x[3,])
# 取單行
x[,4]
# 類別也是一維向量物件
class(x[,4])
# 取不連續的兩行
x[,c(1,3)]
# 用列名取值
x["row3",]
# 行名取值
x[,"col4"]

## ------------------------------------------------------------------------
# 設定drop=FALSE後，傳回單列矩陣
x[3, , drop = F]
# 確認是二維矩陣物件
class(x[3, , drop = F])
# 設定drop=FALSE後，傳回單行矩陣
x[,4, drop = F]
# 確認為二維矩陣物件
class(x[,4, drop = F])

## ------------------------------------------------------------------------
# 資料框以串列的$取值方式取出單一變數的內容(不含變數名稱)
Cars93$Price

## ------------------------------------------------------------------------
# 資料框以串列的一對中括弧取值方式取多個變數
head(Cars93[c('Price', 'AirBags')])
# 資料框以串列的兩對中括弧取值方式取出單一變數內容(不含變數名稱)
Cars93[['DriveTrain']]

## ------------------------------------------------------------------------
# 資料框以矩陣取值方式取出第5筆觀測值
Cars93[5, ]

## 1.3.8 R語言衍生資料物件 ####
## ----warning=FALSE, message=FALSE----------------------------------------
library(DMwR2) # originally library(DMwR)
data(GSPC)
# xts類時間序列物件，列名為時間索引，索引類別為POSIXt
head(GSPC)
str(GSPC)
# 存放數據之矩陣行名(i.e.多變量時間序列變數名)
names(GSPC)
library(xts)
# 以coredata()函數取出核心數據
head(coredata(GSPC))
# 以index()函數取出時間戳記
headtail(index(GSPC))
# xts時間序列物件取值語法
# 以正斜線運算子取出從"2000-02-26"到"2000-03-03"的資料
GSPC["2000-02-26/2000-03-03"]
# 以xtsAttributes()函數取出屬性
xtsAttributes(GSPC)

## ------------------------------------------------------------------------
# 資料期間月數(477 -> 553)
nmonths(GSPC)
# 資料期間季數(159 -> 185)
nquarters(GSPC)
# 資料期間天數(10022 -> 11622)
ndays(GSPC)

## ----warning=FALSE, message=FALSE----------------------------------------
# 擷取資料期間中各週的起迄點
epWks <- endpoints(GSPC, on = "weeks") # 2406起訖點，有2405週！
head(epWks)
# {quantmod}套件內有收盤價擷取函數Cl()
library(quantmod)
# 以period.apply()隱式迴圈函數計算2073(2405)週的收盤價平均值
wksMean <- period.apply(Cl(GSPC),INDEX = epWks,FUN = mean)
class(wksMean)
headtail(wksMean)
# length(wksMean)

## ------------------------------------------------------------------------
# 先產生10022(11622)筆資料的邏輯判斷真假值，儲存為range邏輯值向量
range <- Cl(GSPC) > mean(Cl(GSPC)) + 2.15*sd(Cl(GSPC))
mean(Cl(GSPC)) + 2.15*sd(Cl(GSPC))
# 邏輯值索引取出17筆日資料
GSPC[range]

## 1.5 向量化與隱式迴圈 ------------------------------------------------------------------------
# 方根函數應用到R語言純量
a <- 5
sqrt(a)

## ------------------------------------------------------------------------
b <- c(1.243, 5.654, 2.99)
# 四捨五入函數應用到向量每個元素
round(b)
m <- matrix(runif(12), nrow = 3)
# 對數函數應用到矩陣每個元素
log(m)

## ------------------------------------------------------------------------
# 計算矩陣中所有元素的平均值
mean(m)

## ------------------------------------------------------------------------
# 各橫列(MARGIN = 1表沿橫列)平均值
apply(m, MARGIN = 1, FUN = mean)
# 也可以用rowMeans()
rowMeans(m)
# 各縱行(MARGIN = 2表沿縱行)平均值
apply(m, MARGIN = 2, FUN = mean)
# 也可以用colMeans()
colMeans(m)

## ------------------------------------------------------------------------
# 帶有NA的矩陣
(m <- matrix(c(NA, runif(10), NA), nrow = 3))
# 首末兩列的平均數值為NAs
apply(m, 1, mean)
# `...`的位置傳額外參數到mean()函數中
apply(m, 1, mean, na.rm=TRUE)

## ------------------------------------------------------------------------
# 創建三元素串列
temp <- list(x = c(1,3,5), y = c(2,4,6), z = c("a","b"))
temp
# lapply()逐串列temp之各元素，運用相同函數length()
lapply(temp, FUN = length)
# sapply將回傳結果簡化為具名向量
sapply(temp, FUN = length)

## ------------------------------------------------------------------------
# 長度為3的串列，三個元素類別分別是方陣、矩陣與向量
(firstList <- list(A = matrix(1:16, 4), B = matrix(1:16, 2),
C = 1:5))
# 長度為3的串列，三個元素類別也是方陣、矩陣與向量
(secondList <- list(A = matrix(1:16, 4), B = matrix(1:16, 8),
C = 15:1))
# 以mapply()判斷兩等長串列之對應元素是否完全相同
mapply(FUN = identical, firstList, secondList)

## ------------------------------------------------------------------------
# 自定義匿名函數，注意NROW()與nrow()之異同
simpleFunc <- function(x, y) {NROW(x) + NROW(y)}
# 加總兩串列對應元素之列數和
mapply(FUN=simpleFunc, firstList, secondList)

## ------------------------------------------------------------------------
# 知名的鳶尾花資料集，五個變數為花瓣長寬、花萼長寬與花種
head(iris)

## ----fig.align="center", fig.cap = "\\label{fig:mapply_iris}鳶尾花花萼寬度與花瓣長度散佈情形及分組迴歸直線圖", fig.height=7----
# 花萼寬對花瓣長的散佈圖，pch控制繪圖點字符，注意因子變數轉數值
plot(Sepal.Width ~ Petal.Length, iris, pch =
as.numeric(Species))
# 運用mapply()對split()分組完成的數據，配適模型與畫迴歸直線
regline <- mapply(function(i, x) {abline(lm(Sepal.Width ~
Petal.Length, data = x), lty = i)}, i = 1:3,
x = split(iris, iris$Species))
# 適當位置(4.5, 4.4)上加上圖例說明
legend(4.5, 4.4, levels(iris$Species), cex = 1.5, lty = 1:3)

## 1.6.1 R語言S3類別 ------------------------------------------------------------------------
# 串列創建函數建立物件j
j <- list(name="Joe", salary=55000, union=T)
# 設定物件j之類別為"employee"
class(j) <- "employee"
# 檢視物件j的屬性
attributes(j)

## ------------------------------------------------------------------------
# 注意最下面的"class"屬性
j

## ------------------------------------------------------------------------
# 注意具體方法函數名稱須為：方法.類別
print.employee <- function(wrkr) {
   # 依傳入物件wrkr(i.e. worker)的屬性進行輸出
   cat(wrkr$name,"\n")
   cat("salary",wrkr$salary,"\n")
   cat("union member",wrkr$union,"\n")
}
methods(, "employee")

## ------------------------------------------------------------------------
# 讀者當思考實際呼叫了哪個具體方法
print(j) # print()是泛型函數(a generic function會先判斷物件j的類別)
# print.employee(j) # 直接呼叫print.employee()結果相同！

## ------------------------------------------------------------------------
# 相同亂數種子下結果可重置(reproducible)
set.seed(168)
# 創建weight和height向量
weight <- seq(50, 70, length = 10) + rnorm(10,5,1)
height <- seq(150, 170, length = 10) + rnorm(10,6,2)
# 組成資料框
test <- data.frame(weight, height) # type [tab]
# 建立迴歸模型
test_lm <- lm(weight ~ height, data = test)
# 類別為data.frame
class(test)
# 類別為"lm"
class(test_lm)
# 類別為"ts"
class(AirPassengers) # Univariate Time Series
?AirPassengers
## ----fig.align="center", fig.cap = "\\label{fig:s3_oop}S3泛型函數plot()輸入不同類型物件所繪製的各式圖形", fig.height=8----
# 創建繪圖輸出佈局矩陣
matrix(c(1,1,2:5,6,6), 4, 2, byrow = TRUE)
# 圖面佈局設定
layout(matrix(c(1,1,2:5,6,6), 4, 2, byrow = TRUE))
# 設定繪圖區域邊界，留意重要的繪圖參數設定函數par()
op <- par(mar = rep(2, 4)) # rep()將2重複4次
# 實際呼叫plot.default()
plot(test)
# 實際呼叫plot.lm()
plot(test_lm)
# 實際呼叫plot.ts()
plot(AirPassengers)
# 還原繪圖的預設設定
par(op)
# 還原圖面佈局預設設定
layout(c(1))

## ------------------------------------------------------------------------
# (分身)族繁不及備載
methods(plot)[65:74]
# 查詢多形函數具體方法的使用說明
# ?predict.lm

## 1.7 控制敘述與自訂函數 ####
## 1.7.1 控制敘述 ------------------------------------------------------------------------
x <- c(5,12,13)
# 迴圈敘述關鍵字for
for (n in x) {
  print(n^2)
}

## ------------------------------------------------------------------------
i <- 1
# 迴圈敘述關鍵字while
while (i <= 10){
  i <- i + 4
}
print(i) # 13

## ------------------------------------------------------------------------
# 解的初始值
x <- 2
# 欲求根的函數
f <- x^3 + 2 * x^2 - 7
# 牛頓法容許誤差
tolerance <- 0.000001
while (abs(f) > tolerance) {
  # 求根函數的一階導函數
  f.prime <- 3*x^2 + 4*x
  # 以牛頓法的根逼近公式更新解
  x <- x - f/f.prime
  # 新解的函數值
  f <- x^3 + 2*x^2 - 7
}
# 印出解
x

## ------------------------------------------------------------------------
x <- 2
tolerance <- 0.000001
# 迴圈敘述關鍵字repeat
repeat {
  f <- x^3+2*x^2-7
  if (abs(f) < tolerance) break
  f.prime <- 3*x^2+4*x
  x <- x-f/f.prime
  }
x

## ------------------------------------------------------------------------
grade <- c("C", "C-", "A-", "B", "F")
if (is.character(grade)) {grade <- as.factor(grade)}
if (!is.factor(grade)) {grade <- as.factor(grade)} else
  {print("Grade already is a factor.")}

## 1.7.2 自訂函數 ------------------------------------------------------------------------
# R語言自訂函數，注意關鍵字function，以及三個引數在函數主體
# 如何運用
corplot <- function(x, y, plotit = FALSE) {
    if (plotit == TRUE) plot(x, y)
    # 省略關鍵字return，i.e. return(cor(x, y))
    cor(x, y)
}

## ------------------------------------------------------------------------
# 從連續型均勻分佈Uniform(2, 8)隨機產生u, v亂數
u <- runif(10, 2, 8); v <- runif(10, 2, 8)
# 函數呼叫與傳入引數u與v
corplot(u, v)

## ----fig.align="center", fig.cap = "\\label{fig:plotit_scatterplot}corplot()函數之引數plotit設定為真時繪製的散佈圖"----
# 改變plotit默認值
corplot(u, v, plotit = TRUE)

## ----warning=FALSE, message=FALSE----------------------------------------
# 載入Excel試算表讀檔套件
library(readxl)
newbie <-read_excel("./_data/106_freshmen_final-toR_language.xls")

## ----results="hide"------------------------------------------------------
str(newbie)

## ------------------------------------------------------------------------
# 將選定欄位成批轉換為因子
newbie[-8] <- lapply(newbie[-8], factor)
# 因子或字串變數次數統計
summary(newbie)
# 性別變數有異常，再轉回字串型別做處理
newbie$性別 <- as.character(newbie$性別)
# 前("1男")換為後("男")
newbie$性別 <- gsub("1男", "男", newbie$性別)
# 前("2女")換為後("女")
newbie$性別 <- gsub("2女", "女", newbie$性別)

## ------------------------------------------------------------------------
# 次數分佈確認無誤後再轉為因子
table(newbie$性別)
newbie$性別 <- factor(newbie$性別)

## ------------------------------------------------------------------------
# 將選定欄位成批產生次數分佈表
lapply(newbie[-c(1:4,7,10)], table)

## ------------------------------------------------------------------------
# lapply()加自訂匿名函數成批產生排序後的次數分佈表
lapply(newbie[-c(1:4,7,10)], function(u) {
  # 內圈加外圈的合成函數用法
  sort(table(u), decreasing = TRUE)
})

## ------------------------------------------------------------------------
# 系科學制自訂函數設計
deptByAcaSys <- function(dept="企管系", acasys="四技") {
  TF <- newbie$系所 == dept & newbie$學制 == acasys
  tbl <- newbie[TF,]
  top3 <- head(sort(table(tbl$畢業學校), decreasing = TRUE),
               3)
  bottom3 <- tail(sort(table(tbl$畢業學校), decreasing = TRUE)
                  , 3)
  df <- data.frame(top3 = top3, bottom3 = bottom3)
  names(df) <- c("Top", "TopFreq", "Bottom", "BottomFreq")
  return(df)
}

## ------------------------------------------------------------------------
# 照預設值呼叫函數，仍然要加上成對小括弧
deptByAcaSys()
# 改變函數預設值
deptByAcaSys("會資系", "二技")

## 1.8 資料匯入與匯出 ####
## 1.8.1 R語言資料匯入及匯出 ------------------------------------------------------------------------
# list.files()列出路徑下所有檔名
fnames <- list.files("./_data/C-MAPSS")
# 抓有train開頭的檔名位置
grep("^train", fnames)
# 運用邏輯值索引，形成訓練集檔名向量
(train <- fnames[grep("^train", fnames)])
varNames <- c("unitNo", "cycles","opera1","opera2","opera3","T2","T24","T30","T50","P2","P15","P30","Nf","Nc","epr","Ps30","phi","NRf","NRc","BPR","farB","htBleed","Nf_dmd","PCNfr_dmd","W31","W32")
# 以for迴圈讀取訓練集資料
for (i in 1:length(train)) {
  # 製作各訓練資料物件名稱("train"+流水號)
  myfile <- paste0(unlist(strsplit(train[i], "[_]"))
  [1],i)
  # assign()函數指定字串名稱myfile給依序讀進來的訓練集檔案
  assign(myfile, read.table(paste0("./_data/C-MAPSS/",
  train[i]), header = FALSE, col.names = varNames))
}

# 運用邏輯值索引，形成測試集檔名向量
test <- fnames[grep("^test", fnames)]
# 以for迴圈讀取測試集資料
for (i in 1:length(test)) {
  # 製作各測試資料物件名稱("test"+流水號)
  myfile <- paste0(unlist(strsplit(test[i], "[_]"))[1],i)
  # assign()函數指定字串名稱myfile給依序讀進來的測試集檔案
  assign(myfile, read.table(paste0("./_data/C-MAPSS/",
  test[i]), header = FALSE, col.names = varNames))
}

# varNames <- c("unitNo", "cycles","opera1","opera2","opera3","T2","T24","T30","T50","P2","P15","P30","Nf","Nc","epr","Ps30","phi","NRf","NRc","BPR","farB","htBleed","Nf_dmd","PCNfr_dmd","W31","W32")
# names(train1) <- varNames; names(train2) <- varNames; names(train3) <- varNames; names(train4) <- varNames; names(test1) <- varNames; names(test2) <- varNames; names(test3) <- varNames; names(test4) <- varNames

# 抓有RUL開頭的檔名，形成餘壽資料檔名向量
RUL <- fnames[grep("^RUL", fnames)]
# 以for迴圈讀取餘壽資料
for (i in 1:length(RUL)) {
  # 製作各餘壽資料物件名稱("rul"+流水號)
  myfile <- paste0(tolower(unlist(strsplit(RUL[i],
"[_]"))[1]),i)
  # assign()函數指定字串名稱myfile給依序讀進來的餘壽檔案
  assign(myfile, read.table(paste0("./_data/C-MAPSS/",
  RUL[i]), header = FALSE, col.names = c("RUL")))
}

## ------------------------------------------------------------------------
# 以rm()函數移除工作空間中物件
rm(fnames, i, myfile, RUL, test, train, varNames)
# 儲存工作空間中所有物件為單一RData
# save.image(file = "CMAPSS.RData")
# 下回直接載入所有物件
# load(file = "CMAPSS.RData")

## 1.9 程式除錯與效率監測 ------------------------------------------------------------------------
# 數學函數不能施加在字串上
# log("abc")
# Error in log("abc") : non-numeric argument to mathematical
# function
# log()函數施加在負值上會有警告訊息，
# 告知產生NaNs(Not a number!)
log(-1:2) # 有警告訊息
# library(survival)
# 載入套件時傳回套件{caret}與{survival}中同名物件遮蔽的訊息
library(caret)
# ?cluster # 可看到兩套件的cluster說明文件連結
# 從記憶體中移除caret套件
detach(package:caret)
# 重新載入時不顯示上述訊息
suppressMessages(library(caret))

## ------------------------------------------------------------------------
iter <- 12
# stop()停止程式運行
try(if (iter > 10) stop("too many iterations"))
# Error in try(if (iter > 10) stop("too many iterations")) :
#   too many iterations

## ------------------------------------------------------------------------
# 自訂函數cv()計算向量各元素除以平均值後的標準差
cv <- function(x) {
  sd(x/mean(x))
}
# 天啊！傳入"0"字串既有錯誤又有警告訊息
# cv("0")
# Error in x/mean(x): non-numeric argument to binary operator
# In addition: Warning message:
# In mean.default(x) :
# Error in x/mean(x): non-numeric argument to binary operator

## ------------------------------------------------------------------------
# 查看呼叫堆疊(call stack)
# traceback()
# 4: is.data.frame(x)
# 3: var(if (is.vector(x) || is.factor(x)) x else
# as.double(x), na.rm = na.rm)
# 2: sd(x/mean(x)) at #2
# 1: cv("0")

## ------------------------------------------------------------------------
# 先測試分母，結果為NA，但有下面的警告訊息！
mean("0")

## ------------------------------------------------------------------------
# 分母是警告訊息的來源，分子除以分母才是錯誤訊息的來源
# "0" / mean("0")
# Error in "0"/mean("0") : non-numeric argument to binary
# operator
# In addition: Warning message:
# In mean.default("0") : argument is not numeric or logical:
# returning NA

## ------------------------------------------------------------------------
# "0"/NA
# Error in "0"/NA : non-numeric argument to binary operator

## ------------------------------------------------------------------------
# 既無錯誤亦無警告
0/NA

## ------------------------------------------------------------------------
# 加入合理性檢查(sanity check)敘述修正原函數
cv <- function(x) {
  # sanity check
  stopifnot(is.numeric(x))
  sd(x / mean(x))
}
# 傳入"0"直接停止程式運行且報錯(stopifnot())
# cv("0")
# Error: is.numeric(x) is not TRUE

## ------------------------------------------------------------------------
# cat()印出中間結果，尤其是mean(x)，因為它在分母
cv <- function(x) {
  cat("In cv, x=", x, "\n")
  cat("mean(x)=", mean(x), "\n")
  sd(x/mean(x))
}
cv(0:3)

## ------------------------------------------------------------------------
# 函數主體首行加入browser()，以進入偵錯模式
cv <- function(x) {
  browser()
  cat("In cv, x=", x, "\n")
  cat("mean(x)=", mean(x), "\n")
  sd(x/mean(x))
}
cv(0:3)

## ------------------------------------------------------------------------
# debug()與undebug()偵錯模式示例
# cv <- function(x) {
#   cat("In cv, x=", x, "\n")
#   cat("mean(x)=", mean(x), "\n")
#   y <- mean(x)
#   sd(x/y)
# }
# debug(cv)
# cv(0:3)
# undebug(cv) # 記得離開偵測模式
# cv(0:3)

## ------------------------------------------------------------------------
# 輸入的向量中，奇數元素的個數(%%表除法運算後取餘數)
oddcount <- function(x) {return(sum(x %% 2 == 1))}
# 隨機從1到1000000中置回抽樣取出100000個整數
x <- sample(1:1000000, 100000, replace = T)
system.time(oddcount(x))

