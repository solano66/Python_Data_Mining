########################################################## Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the IDS and CADS(資訊與決策科學研究所暨資料科學應用研究中心), NTUB(國立臺北商業大學); the CARS(中華R軟體學會); and the DSBA(臺灣資料科學與商業應用協會)
#######################################################
### Data sets: test.sav, SampleData_103.csv, contribution.csv, songPoem_win.csv, spq.csv, germancredit.csv 
### Notes: This code is provided without warranty.
### Except where otherwise noted, content on this R script is licensed under a Creative Commons BY-NC-ND license. (除非另有註明，本文件之內容皆採用創用CC 姓名標示-非商業性-禁止改作 授權方式)

### Preliminary 0 RStudio four panes introduction #### (編碼、執行、環境與套件管理，執行結果可能出現在後三個panes中)
### 1. File -> Reopen with Encoding -> UTF-8 (for MS Windows 中中文顯示)
### 2. Tools -> Global Options -> Code -> Soft-wrap R source files (程式碼自動換行)
### 3. JGR + Deducer, Rcmdr, Rattle GUIs demos (for 初學者編碼不熟悉，透過圖形化使用者介面逐步上手)

### Supplement 0: Masking ####
library(Hmisc)
describe(iris)
library(psych)
describe(iris)
describe
Hmisc::describe(iris)

### Supplement 1: Vectorization and Recycling ####
x <- 1:10 # "<-" is assignment operator
x
x^2 # vector in (x) and vector out (x^2)

x > 6

### Supplement 2: Packages Management ####
search() # 檢查RAM載入了哪些套件
installed.packages() # 檢查本機硬碟下載了哪些套件

str(installed.packages())
colnames(installed.packages())

.libPaths()
library()

### Base Packages
search()
installed.packages() # eval=FALSE
search() # 有看到stats
?hclust
hc <- hclust(dist(USArrests), "ave")
plot(hc, hang = -1)

### Recommended Packages
grep("lattice", search(), value=TRUE) # 應該抓不到lattice
grep("lattice", rownames(installed.packages()), value=TRUE)
library(lattice)
dotplot(variety ~ yield | year * site, data=barley)

### Contributed Packages
data(team.batting.00to08) # Warning message: In data(team.batting.00to08) : data set ‘team.batting.00to08’ not found
str(team.batting.00to08) # Error in str(team.batting.00to08) : object 'team.batting.00to08' not found
library(nutshell) # Error in library(nutshell) : there is no package called ‘nutshell’
search() # 未載入nutshell
grep("nutshell", rownames(installed.packages()), value=TRUE) # 未下載nutshell
install.packages("nutshell") # 可透過GUI完成
library(nutshell)
data(team.batting.00to08)
str(team.batting.00to08[,-(10:13)])

### Supplement 3: Finding help ####
help.start() # Manuals, Reference, and Miscellaneous Material

help('plot') # function name already known (or help(plot))
?plot # an alias (別名) of help

help.search('plot') # function name unknown, search string from Vignette Names, Code Demonstrations, and Help Pages (titles and keywords)擴大搜尋範圍
??plot # an alias of help.search

apropos('plot') # function name with 'plot'

find('plot') # It is a different user interface to the same task. simple.words: logical; if TRUE, the what argument is only searched as whole word.
find('plot', simple.words = FALSE) # look for packages with function 'plot'

### Supplement 4: Workspace and Environment ####
search()
objects()
ls()
getwd()
setwd("~")
getwd()

### Supplement 5: Load SPSS and Excel files ####
library(foreign)

db <- read.spss("test.sav", to.data.frame=TRUE)
head(db)
str(db)

library(readxl)
excel_sheets(system.file('extdata/datasets.xlsx', package='readxl'))

# To load all sheets in a workbook, use lapply
(filepath <- system.file('extdata/datasets.xls', package='readxl'))
x <- lapply(excel_sheets(filepath), read_excel, path=filepath)
str(x)
x[1]
class(x[1])
x[[1]]
class(x[[1]])

read_excel(filepath) # sheet預設為1
read_excel(filepath, 2)
read_excel(filepath, 'mtcars') # same as above

### Read a bigger csv file ####
system.time(trns02 <- read.csv(file="SampleData_103.csv", header=TRUE, fileEncoding = "Big-5"))
head(trns02)

library(data.table) # for function "fread"
system.time(trns03 <- fread("SampleData_103.csv", header=TRUE))
head(trns03)
names(trns03) <- names(trns02)

save(trns03, file="/Users/vince/cstsouMac/IndustryLink/IOT/trns03.RData") # "C:/Users/..."
system.time(load("/Users/vince/cstsouMac/IndustryLink/IOT/trns03.RData"))
head(trns03)

### Supplement 6: S3 - object-oriented programming & generic functions #####
set.seed(168)
layout(matrix(c(1,1,2:5,6,6),4,2, byrow=TRUE))
weight <- seq(50, 70, length=10) + rnorm(10,5,1)
height <- seq(150, 170, length=10) + rnorm(10,6,2)
test <- data.frame(weight, height) # type [tab]
test.lm <- lm(weight ~ height, data=test) # outcome(因) ~ predictor(自1 + 自2...), lm: linear models
class(test) # "data.frame"
class(test.lm) # "lm", same as the function name
class(AirPassengers) # "ts"
plot(test)
plot(test.lm)
plot(AirPassengers)
layout(c(1))
methods(plot)

### Slides Codes start from here !!!
### Data Objects in packages {UsingR} and {datasets} ####
### 基本概念：關心變數型別、資料(存放)結構(by str())與其類別(by class())
library(UsingR) # for datasets bumpers, aid, firstchi, and crime
data(firstchi)
firstchi
help(firstchi) # An alias of help() is ?
class(firstchi)
names(firstchi) # NULL means without names

bumpers # a named vector
help(bumpers)
class(bumpers)
names(bumpers)

crime
help(crime)
class(crime)
names(crime)
dimnames(crime) # What is the difference to names()?

Harman23.cor
help(Harman23.cor)
class(Harman23.cor)
names(Harman23.cor)

Harman23.cor$cov
class(Harman23.cor$cov)
dimnames(Harman23.cor$cov) # NOT names !

Titanic;sum(Titanic)
help(Titanic)
class(Titanic) # "table" sometimes means "array"
dimnames(Titanic)
ftable(Titanic) # flatten the table
sum(ftable(Titanic))
sum(Titanic)

##### Accessing various data structures #####
# vector indexing
x <- 20:16 # <- equivalent to =
x
# 元素設定名稱
names(x) <- c("1st", "2nd", "3rd", "4th", "5th")
x
# 單一位置取值
x[4]
x[-4] # "-" means drop out
# 單一名稱取值(if具名向量)
x["4th"]
# 連續位置範圍取值
x[1:4]
x[-(1:4)]
# 位置間隔取值(注意位置的錯置)
x[c(1,4,2)]
# 位置重覆取值
x[c(1,2,2,3,3,3,4,4,4,4)]
# 多重名稱取值(if具名向量)
x[c("1st","3rd")]
# 真假邏輯取值(邏輯值索引)
x[c(T,T,F,F,F)]
x[x > 18] # x > 18 means x > c(18, 18, 18, 18, 18, 18)
x[x > 16 & x < 19]
# 二元運算子 %in% 回傳值亦為真假邏輯值
x[x %in% c(16, 18, 20)]

# list indexing
x <- list(one=1, two=2, three=3, four=4, five=5) # key1=value1, key2=value2
x
x[4] # it's a sublist!
x[[4]] # it's a vector of length 1
x["four"] # same as x[4]
x$four # same as x[[4]]

## Another example: create a list first
g <- "My First List"
h <- c(25, 26, 18, 39)
j <- matrix(1:10, nrow=5, byrow=T)
k <- c("one", "two", "three")
mylist <- list(title=g, ages=h, j, k)
mylist

# Different list subsetting
mylist[[2]] # get the element(s) of a list
class(mylist[[2]]) # a one-dimensional numeric vector
mylist[["ages"]] # another way to subsetting a list
mylist$ages
mylist[[1:2]] # Error in mylist[[1:2]] : subscript out of bounds

mylist[2] # get a part of a list, same as below (a sublist)
class(mylist[2]) # a list
mylist[1:2]

# matrix indexing
x <- matrix(1:12, nrow=3)
x
dimnames(x) <- list(paste("row", 1:3, sep='-'),paste("col", 1:4, sep='')) # sep: separator
x[3,4]
x[3,]
x[,4]
x[,c(1,3)]
x["row3",]
x[,"col4"]

# logical indexing
x <- 1:10
x[c(T,T,T,F,F,F,F,F,F,F)] # T: keep it; F: drop out
x[c(rep(T,3), rep(F,7))]
x <= 3
x[x <= 3]

##### apply, lapply, and sapply function  for replacing the for loops #####
(m <- matrix(1:10, nrow=5, byrow=T))
apply(X=m, MARGIN=1, FUN=mean) # 逐列計算mean
apply(m, 2, mean) # 逐行計算mean
?apply
(temp <- list(x = c(1,3,5), y = c(2,4,6), z = c("a","b")))
temp
lapply(temp, length) # list in list ***out*** (逐***元素***套用後面的函數)
class(lapply(temp, length)) # "list"
sapply(temp, length) # s means ***simplify*** your output if possible!
class(sapply(temp, length)) # "integer"

### Supplement 7: mapply ####
(firstList <- list(A = matrix(1:16, 4), B = matrix(1:16, 2), C = 1:5))
(secondList <- list(A = matrix(1:16, 4), B = matrix(1:16, 8), C = 15:1))

mapply(identical, firstList, secondList) #
?mapply
simpleFunc <- function(x, y)
{
  NROW(x) + NROW(y)
  
}

mapply(simpleFunc, firstList, secondList)

### Practice 0: (a) 以一行R指令建立五個元素分別為"1-a-A"、"2-b-B"、"3-c-C"、"4-d-D"與"5-e-E"的字串向量(Hint: 運用mapply與paste函數)。(b) 接著利用套件{C50}中建立客戶流失模型的資料集churn，請問該如何將之載入環境中？如何驗證churnTest與churnTrain的屬性/變數/欄位名稱是相同的？最後，做出churnTrain各屬性的摘要統計值，並繪製其直方圖。####
paste(c(1,2,3), c("a","b","c"), sep="-")
mapply(paste, 1:5, letters[1:5], LETTERS[1:5], sep="-")

### Part 1 base graphics {graphics} ####
attach(mtcars)
plot(wt, mpg) # high-level plotting
abline(lm(mpg~wt)) #low-level plotting according to the results of linear model
title("Regression of MPG on Weight") #low-level plotting

## Supplement: 互動式標出資料點
identify(wt, mpg, n=3)
# identify(locator(), n=1)
detach(mtcars)

? identify # interactive plot

### {graphics}: Output graphics to pdf ####
pdf("mygraph.pdf")
attach(mtcars)
plot(wt, mpg)
abline(lm(mpg~wt))
title("Regression of MPG on Weight")
detach(mtcars)
dev.off()

### {graphics}: A simple example ####
dose  <- c(20, 30, 40, 45, 60)
drugA <- c(16, 20, 27, 40, 60)
drugB <- c(15, 18, 25, 31, 40)
plot(dose, drugA, type="b") #"b": both, points and line
help(plot) # view other options

### Supplement: other types ####
plot(dose, drugA, type="p")
plot(dose, drugA, type="l")
plot(dose, drugA, type="c")
plot(dose, drugA, type="o")
plot(dose, drugA, type="h")
plot(dose, drugA, type="s")
plot(dose, drugA, type="S")
plot(dose, drugA, type="n")
text(dose, drugA, labels=paste(letters[1:5], 1:5, sep=""))

### {graphics}: Graphical parameters ####
opar <- par(no.readonly=TRUE) #存出預設設定, 以供還原
par(lty=2, pch=17)
plot(dose, drugA, type="b") #"b": both, points and line
par(opar) #還原

plot(dose, drugA, type="b", lty=2, pch=17) #同上

### {graphics}: 圖形參數分類 ####
par("mar")

?par
par(mar=c(0,4,0,2))
par("mar")

### {graphics}: 高階繪圖、低階繪圖與圖形參數pch, col, cex, lty, lwd ####
plot(1,1,xlim=c(1,7.5),ylim=c(0,5),type="n")
points(1:7,rep(4.5,7),cex=1:7,col=1:7,pch=0:6) # cex (charactoer expansion), pch: 0:6
text(1:7,rep(3.5,7),labels=paste(0:6),cex=1:7,col=1:7)
points(1:7,rep(2,7),pch=(0:6)+7) # pch: 7:13
text((1:7)+.25,rep(2,7),paste((0:6)+7))
points(1:7,rep(1,7),pch=(0:6)+14) # pch: 14:20
text((1:7)+.25,rep(1,7),paste((0:6)+14))

plot.new()
legend(locator(1), as.character(0:25), pch = 0:25)

plot.new()
legend(locator(1), as.character(0:25), lty = 0:25)

plot.new()
legend(locator(1), as.character(0:25), lwd = 0:25)

plot(dose, drugA, type="b", col="red", lty=2, pch=2, lwd=2, main="Clinical Trials for Drug A", sub="This is hypothetical data", xlab="Dosage", ylab="Drug Response", xlim=c(0, 60), ylim=c(0, 70)) # col: color, lwd: line width

par(mar=c(4,4,2,2))
x=0:5
y=pi*x^2
plot(x,y,xlab="Radius",ylab=expression(Area==pi*r^2))

### {graphics}: Comparing drug A and drug B response by dose ####
dose  <- c(20, 30, 40, 45, 60)
drugA <- c(16, 20, 27, 40, 60)
drugB <- c(15, 18, 25, 31, 40)
opar <- par(no.readonly=TRUE)
par(lwd=2, cex=1.5, font.lab=2)
plot(dose, drugA, type="b", pch=15, lty=1, col="red", ylim=c(0, 60), main="Drug A vs. Drug B", xlab="Drug Dosage", ylab="Drug Response")
lines(dose, drugB, type="b", pch=17, lty=2, col="blue")
abline(h=c(30), lwd=1.5, lty=2, col="gray") #abline(h=?) 水平線 vs v=? 垂直線
library(Hmisc)
minor.tick(nx=3, ny=3, tick.ratio=0.5)
legend("topleft", inset=.05, title="Drug Type", c("A","B"), lty=c(1, 2), pch=c(15, 17), col=c("red", "blue")) # attention to inset!
par(opar)

### {graphics}: Text annotations ####
attach(mtcars)
plot(wt, mpg, main="Mileage vs. Car Weight", xlab="Weight", ylab="Mileage", pch=18, col="blue")
text(wt, mpg, row.names(mtcars), cex=0.6, pos=4, col="red") #pos=4 1,2,3,4 means add labels the bottom, left, up, right
detach(mtcars)

### {graphics}: Combining graphs ####
attach(mtcars)
opar <- par(no.readonly=TRUE)
par(mfcol=c(2,2)) #par(mfcol=c(2,2))
plot(wt,mpg, main="Scatterplot of wt vs. mpg")
plot(wt,disp, main="Scatterplot of wt vs. disp")
hist(wt, main="Histogram of wt")
boxplot(wt, main="Boxplot of wt")
par(opar)
detach(mtcars)

### {graphics}: Combining graphs - fine placement of figures in a graph ####
opar <- par(no.readonly=TRUE)
par(fig=c(0, 0.8, 0, 0.8))
plot(mtcars$wt, mtcars$mpg, xlab="Car Weight", ylab="Miles Per Gallon")
par(fig=c(0, 0.8, 0.55, 1), new=TRUE)
boxplot(mtcars$wt, horizontal=TRUE, axes=FALSE)
par(fig=c(0.65, 1, 0, 0.8), new=TRUE)
boxplot(mtcars$mpg, axes=FALSE)
mtext("Enhanced Scatterplot", side=3, outer=TRUE, line=-3)
par(opar)

### 圖形客製化示例 ####
p <- par(mfrow=c(2,2))
with(iris, plot(Sepal.Length, Sepal.Width, col=Species, pch=as.numeric(Species), las=0)
)
with(iris, plot(Sepal.Length, Sepal.Width, col=Species, pch=as.numeric(Species), las=1)
)
with(iris, plot(Sepal.Length, Sepal.Width, col=Species, pch=as.numeric(Species), las=2)
)
with(iris, plot(Sepal.Length, Sepal.Width, col=Species, pch=as.numeric(Species), las=3)
)
par(op)

# {graphics}: 各式圖形 ####
# 繪圖前資料理解
data(iris)
dim(iris)

names(iris)

str(iris)

attributes(iris)

head(iris)

tail(iris)

# 單變數繪圖-類別型 ####
# 圓餅圖
table(iris$Species)
pie(table(iris$Species))
# Error in pie(iris$Species) : 'x' values must be positive.
pie(as.numeric(iris$Species))

# 3D圓餅圖
library(plotrix)
?pie3D
pie3D(table(iris$Species), labels = c("Setosa", "Versicolor", "Virginica"))
levels(iris$Species)

# 扇形圖
slices <- c(10, 12, 4, 16, 8)
country <- c('US', 'UK', 'Australia', 'Germany', 'France')
?fan.plot
fan.plot(slices, labels = country, main = 'Fan Plor')

# 單變數繪圖-數值型 ####
data(iris)
?hist
hist(iris$Sepal.Length, xlab = 'Speal Length', main = '花萼長度直方圖')

plot(density(iris$Sepal.Length))

library(MASS)
Cars93
library(car)
qqPlot(Cars93$MPG.city)

# 待補 ####
# 多變數繪圖前置工作 ####

# 雙變數繪圖-兩類別變數 ####

# 雙變數繪圖-兩數值變數 ####

# 多變數繪圖-多(2+)數值變數 ####

# 多變數繪圖-三數值變數 ####

# 雙變數繪圖-混合型變數 ####

# 多變數繪圖-混合型變數 ####

# 多變數繪圖-其它 ####

# 三變數互動式繪圖 ####
library(rgl)
plot3d(iris$Petal.Width, iris$Sepal.Length, iris$Sepal.Width)

### Practice 1: (a) 以vcd套件中的關節炎資料集Arthritis之Improved與Treatment變數，繪製堆疊長條圖與並排長條圖。(b) 將mtcars資料集中的mpg變數繪製直方圖(注意y軸尺度)，並配上密度曲線與橫軸計數刻度。(c) 將mtcars資料集中的mpg變數依am與cyl兩變數繪製並排盒鬚圖。 ####

### Part 2 lattice ####
d <- data.frame(x=c(0:9), y=c(1:10), z=c(rep(c("a","b"), times=5)))
d
library(lattice)
xyplot(y~x, data=d)
#library(lattice)
xyplot(y~x|z, data=d) #多面板繪圖
xyplot(y~x, groups=z, data=d,auto.key=TRUE) # 疊加圖, auto.key=TRUE 加圖例

# Case for {lattice}
library(nutshell) # 此套件已下架，請自行google連結到Archive中下載.tar.gz或是.zip後，再從本機硬碟中安裝之(nutshell.bbdb和nutshell.audioscrobbler等相依套件也須一併下載安裝)
data(births2006.smpl) # births2006.smpl <- read.csv('births2006.smpl.csv')
births2006.smpl[1:3,]
dim(births2006.smpl)
library(lattice)
births.dow=table(births2006.smpl$DOB_WK)
births.dow
barchart(births.dow,ylab="Day of Week",col="pink") # 一維列聯表(一維次數分配表)之視覺化
dob.dm.tbl=table(WK=births2006.smpl$DOB_WK,MM=births2006.smpl$DMETH_REC)
dob.dm.tbl=dob.dm.tbl[,-2]
dob.dm.tbl
barchart(dob.dm.tbl,ylab="Day of Week",auto.key=TRUE) # 二維列聯表(二維次數分配表)之視覺化
barchart(dob.dm.tbl,stack=FALSE, ylab="Day of Week",auto.key=TRUE) # side-by-side barchart
barchart(dob.dm.tbl,horizontal=FALSE,groups=FALSE,xlab="Day of Week",col="blue") # horizontal=FALSE: 直立式; groups=FALSE: 分面繪
dob.dm.tbl.alt <- table(WEEK=births2006.smpl$DOB_WK, MONTH=births2006.smpl$DOB_MM,METHOD=births2006.smpl$DMETH_REC) # Three-way contingency table, 三維列聯表
dob.dm.tbl.alt <- dob.dm.tbl.alt[,,-2] # remove METHOD = Unknown
dotplot(dob.dm.tbl.alt, auto.key=TRUE, layout=c(3,4)) # 克里夫蘭點圖 layout=c(3,4)) col,row
histogram(~DBWT|DPLURAL,data=births2006.smpl,layout=c(1,5),col="black") # Higher Plural Births, less Birth Weight
histogram(~DBWT|DMETH_REC,data=births2006.smpl,layout=c(1,3),col="black")
densityplot(~DBWT|DPLURAL,data=births2006.smpl,layout=c(1,5),plot.points=FALSE,col="black") #plot.points: draw datapoint
densityplot(~DBWT,groups=DPLURAL,data=births2006.smpl,plot.points=FALSE, auto.key=TRUE) # 疊加圖
dotplot(~DBWT|DPLURAL, data=births2006.smpl, layout=c(1,5), plot.points=FALSE, col="black")
stripplot(~DBWT, data=births2006.smpl, subset=(DPLURAL=="5 Quintuplet or higher" | DPLURAL=="4 Quadruplet"), jitter.data=TRUE) #挑出點數較少的數據, jitter 晃動點以免overplot

nrow(subset(births2006.smpl, subset=(DPLURAL=="5 Quintuplet or higher" | DPLURAL=="4 Quadruplet"))) # 44 observaton

qqmath(~DBWT|DPLURAL, data=births2006.smpl[sample(1:nrow(births2006.smpl), 50000),], pch=19, cex=0.25, subset=(DPLURAL != "5 Quintuplet or higher")) # 單變量分位數圖, distribution = qnorm
xyplot(DBWT~WTGAIN|DPLURAL,data=births2006.smpl,layout=c(1,5),col="black") # y ~ x
smoothScatter(births2006.smpl$WTGAIN,births2006.smpl$DBWT) # x, y
? smoothScatter
bwplot(DBWT~factor(APGAR5)|factor(SEX),data=births2006.smpl,xlab="AGPAR5")
bwplot(DBWT~factor(DOB_WK),data=births2006.smpl,xlab="Day of Week")
new=births2006.smpl[births2006.smpl$ESTGEST != 99,]
t51=table(new$ESTGEST)
t51
t6=tapply(new$DBWT,INDEX=list(cut(new$WTGAIN,breaks=10),cut(new$ESTGEST,breaks=10)),FUN=mean,na.rm=TRUE)
t6

(Z <- stats::rnorm(10000))
-6:6
cut(Z, breaks = -6:6)
table(cut(Z, breaks = -6:6))

levelplot(t6, scales = list(x = list(rot = 90)), xlab = "WTGAIN", ylab = "ESTGEST")
contourplot(t6,scales = list(x = list(rot = 90)))

### Practice 2: 請以lattice畫iris中不同品種下 Petal.Length 的盒鬚圖.(分面板與不分面板(疊加圖)各一張) ####
data(iris)
str(iris)
bwplot(Petal.Length~Species,data=iris,xlab="Species",ylab="Petal.Length")
bwplot(~Petal.Length|Species,layout=c(1, 3), data=iris,xlab="Petal.Length",ylab="Species")

### Practice 3: 請以 lattice 畫 nutshell 套件中 sanfrancisco.home.sales 資料集, 其房屋面積 squarefeet 與房價 price 的散佈圖, 請問兩屬性有何關係? 後續可做那些條件式繪圖? ####

library(nutshell)
data(sanfrancisco.home.sales)
str(sanfrancisco.home.sales)

xyplot(price~squarefeet, data=sanfrancisco.home.sales)
xyplot(price~squarefeet|month, layout=c(3,6),data=sanfrancisco.home.sales)

### Case 1: Exploratory Data Analysis: Alumni Donation Case ####
don <- read.csv(file.choose()) # select "contribution.csv"
str(don)
don[1:5,] # or head(don, 5)
table(don$Class.Year) # Which one is the smallest cohort? Why? Beacuse of smaller class sizes in the past and deaths of older alumni
library(lattice)
barchart(table(don$Class.Year),horizontal=FALSE,xlab="Class Year",col="black") # Visualizing above table

don$TGiving <- don$FY00Giving + don$FY01Giving + don$FY02Giving + don$FY03Giving + don$FY04Giving # Total contributions for 2000-2004 are calculated for each graduate
str(don)

mean(don$TGiving)
sd(don$TGiving) # it's so huge !
quantile(don$TGiving,probs=seq(0,1,0.05)) # 90% gave $1050 or less, and more than 30% gave nothing
quantile(don$TGiving,probs=seq(0.95,1,0.01)) # Look into the details (with step size 0.01) and find that only 3% gave more than $5000. The largest contribution was $ 171870.06

hist(don$TGiving) # Visualizing total givings (histogram of TGiving)
hist(don$TGiving[don$TGiving!=0 & don$TGiving <= 1000]) # Visualizing total givings excluding those gave nothing and focusing on contributions less than 1000 (histogram of TGiving truncated)

## or, if you want to achieve the above histogram slower in two steps
## ff1=don$TGiving[don$TGiving!=0]
## ff1
## ff2=ff1[ff1<=1000]
## ff2
## hist(ff2,main=paste("Histogram of TGivingTrunc"),xlab="TGivingTrunc")

boxplot(don$TGiving,horizontal=TRUE,xlab="Total Contribution") # Visualizing total givings by boxplot
boxplot(don$TGiving,outline=FALSE,horizontal=TRUE,xlab="Total Contribution") # Visualizing total givings by boxplot without plotting outliers. Attention please ! The x-axis ranges from $0 to $1000

ddd=don[don$TGiving>=30000,] # Alumni with more than $30,000 contributions
ddd1=ddd[,c('Gender', 'Class.Year', 'Marital.Status', 'Major', 'Next.Degree', 'TGiving')]
ddd1
ddd1[order(ddd1$TGiving,decreasing=TRUE),] # Ordering whole data frame by TGiving to identify the top donor has a mathematics-physics double major with no advanced degree

# It is important to know who is contributing, so box plots of total 5-year donation for the class year, gender, marital status, and attendance at a event
boxplot(TGiving~Class.Year,data=don,outline=FALSE) # The outliers are not drawn. Older alumni have higher life earnings.
boxplot(TGiving~Class.Year,data=don) # Which one is more informative?
boxplot(TGiving~Gender,data=don,outline=FALSE) # Dont' say I have sex discrimination!
boxplot(TGiving~Marital.Status,data=don,outline=FALSE) # Single and divorced alumni give less.
boxplot(TGiving~AttendenceEvent,data=don,outline=FALSE) # Attendance at a foundation-sponsored event certainly helps

# Boxplots of total giving against the alumni's major and second degree.
t4=tapply(don$TGiving,don$Major,mean,na.rm=TRUE) # Mean TGiving across majors
t4;length(t4)
t5=table(don$Major)
t5;length(t5)
t6=cbind(t4,t5)
t7=t6[t6[,2]>10,] # Only show those majors with frequencies exceeding a certain threshold
t7[order(t7[,1],decreasing=TRUE),] # Show TGiving across majors by decreasing order
barchart(sort(t7[,1]),col="black", xlab='Mean TGiving', main="TGiving against Major")


t4=tapply(don$TGiving,don$Next.Degree,mean,na.rm=TRUE)
t4;length(t4)
t5=table(don$Next.Degree)
t5;length(t5)
t6=cbind(t4,t5)
t7=t6[t6[,2]>10,] # Only show those majors with frequencies exceeding a certain threshold
t7[order(t7[,1],decreasing=TRUE),] # Show TGiving across majors by decreasing order
barchart(sort(t7[,1]),col="black", xlab='Mean TGiving', main="TGiving against Next.Degree")

# The distribution of 5-year giving among alumni who gave $1 - $1000, stratified according to year of graduation.
densityplot(~TGiving|factor(Class.Year), data=don[don$TGiving<=1000,][don[don$TGiving<=1000,]$TGiving>0,], plot.points=FALSE,col="black")

# Calculating the total 5-year donations for the five graduation cohorts
t11=tapply(don$TGiving,don$Class.Year,FUN=sum,na.rm=TRUE)
t11
barplot(t11,ylab="Total Donation",xlab="Class Year") # Older alumni more donation

# Annual contributions (2000-2004) of the five graduation classes
# barchart(tapply(don$FY04Giving,don$Class.Year,FUN=sum,na.rm=TRUE),horizontal=FALSE,ylim=c(0,225000),col="black")
# barchart(tapply(don$FY03Giving,don$Class.Year,FUN=sum,na.rm=TRUE),horizontal= FALSE,ylim=c(0,225000),col="black")
# barchart(tapply(don$FY02Giving,don$Class.Year,FUN=sum,na.rm=TRUE),horizontal= FALSE,ylim=c(0,225000),col="black")
# barchart(tapply(don$FY01Giving,don$Class.Year,FUN=sum,na.rm=TRUE),horizontal= FALSE,ylim=c(0,225000),col="black") # The year 2001 was the best because of some very large contributions from the 1957 cohort. 
# barchart(tapply(don$FY00Giving,don$Class.Year,FUN=sum,na.rm=TRUE),horizontal= FALSE,ylim=c(0,225000),col="black")

op <- par(mfrow=c(2,3))
barplot(tapply(don$FY04Giving,don$Class.Year,FUN=sum,na.rm=TRUE),
        ylim=c(0,225000),col="black",main="2004",las=3)
barplot(tapply(don$FY03Giving,don$Class.Year,FUN=sum,na.rm=TRUE),
        ylim=c(0,225000),col="black",main="2003",las=3)
barplot(tapply(don$FY02Giving,don$Class.Year,FUN=sum,na.rm=TRUE),
        ylim=c(0,225000),col="black",main="2002",las=3)
barplot(tapply(don$FY01Giving,don$Class.Year,FUN=sum,na.rm=TRUE),
        ylim=c(0,225000),col="black",main="2001",las=3)
barplot(tapply(don$FY00Giving,don$Class.Year,FUN=sum,na.rm=TRUE),
        ylim=c(0,225000),col="black",main="2000",las=3)
par(op)

# Computing and analyzing the numbers and proportions of individuals who contributed (分析捐款人數與比例)
summary(don$TGiving)
sort(don$TGiving)[1:425] # $0 -> $5
don$TGivingIND=cut(don$TGiving,c(-1,0.5,10000000),labels=FALSE)-1 # (did, did not): from (2,1) to (1,0)
mean(don$TGivingIND) # About 66% of all alumni contribute
t5=table(don$TGivingIND,don$Class.Year)
t5 # Donation or not number versus Class year
barplot(t5,beside=TRUE)
mosaicplot(factor(don$Class.Year)~factor(don$TGivingIND)) # Pay your attention to factor
t50=tapply(don$TGivingIND,don$Class.Year,FUN=mean,na.rm=TRUE)
t50 # Donation proportion across Class year
barchart(t50,horizontal=FALSE,col="black",xlab='Class Year', main="Donation Proportion")

don$FY04GivingIND=cut(don$FY04Giving,c(-1,0.5,10000000),labels=FALSE)-1
t51=tapply(don$FY04GivingIND,don$Class.Year,FUN=mean,na.rm=TRUE)
t51 # Donation proportion of 2004 (the lastest year) across Class year
barchart(t51,horizontal=FALSE,col="black",xlab='Class Year', main="Donation Proportion of FY04Giving")

# Exploring the relationship between the alumni contributions among the 5 years (五年捐款金額關係)
Data=data.frame(don$FY04Giving,don$FY03Giving,don$FY02Giving,don$FY01Giving,don$FY00Giving) # Collate the five year contribution data
correlation=cor(Data)
correlation
plot(Data)
library(corrplot)  
corrplot(correlation)
corrplot(correlation, method="pie", tl.col="black", tl.srt=45)

col <- colorRampPalette(c('#BB4444', '#EE9988', '#FFFFFF', '#77AADD', '#4477AA'))
corrplot(correlation, method="shade", shade.col=NA, tl.col="black", tl.srt=45, col=col(200), addCoef.col="black", order='AOE') # addCoef.col=Color of coefficients added on the graph, order={"AOE"|"FPC"|"hclust"}: the ordering method of the correlation matrix

mosaicplot(factor(don$Gender)~factor(don$TGivingIND)) # Same for men and women
mosaicplot(factor(don$Marital.Status)~factor(don$TGivingIND)) # Married are more likely to contribute
mosaicplot(factor(don$AttendenceEvent)~factor(don$TGivingIND)) # Attendence are more likely to contribute and more than half have attended such a meeting
# t2=table(factor(don$Marital.Status),factor(don$TGivingIND))
# mosaicplot(t2)

t2=table(factor(don$Marital.Status),factor(don$TGivingIND),factor(don$AttendenceEvent)) # Three way contingency table
t2
op <- par(mfrow=c(1,2))
mosaicplot(t2[,,1])
mosaicplot(t2[,,2]) # The likelihood of giving increases with attendance, but the relative proportions of giving across the marital status groups are fairly similar. (Main effect of attendance exists, but there is not much of an interaction effect.)
par(op)

### Part 3: ggplot2 ####
library(ggplot2)

str(diamonds) #五萬多筆
data(diamonds)
help(diamonds)
set.seed(1410)
dsmall <- diamonds[sample(nrow(diamonds), 1000), ]

### ggplot: scatterplot
p <- ggplot(data=dsmall, mapping=aes(x=carat, y=price, color=color)) + geom_point(size=5) # base layer (data, aesthetic mapping) + geometric layer
p
### ggplot: scatterplot + regression line
# se (standard error): display confidence interval around smooth ? (TRUE by default) level: level of confidence interval to use (0.95 by default)
p <- ggplot(dsmall, aes(carat, price)) + geom_point() + geom_smooth(method="lm", se=TRUE) # base layer (data, mapping) + geometric layer + statistical transormation layer
p

p <- ggplot(dsmall, aes(carat, price)) + geom_point() + geom_smooth(method="lm", se=TRUE)
p
?stat_smooth

p <- ggplot(dsmall, aes(carat, price)) + geom_point() + geom_smooth(method="lm") # 'se' is defaulted to TRUE
p
### ggplot2 scatter plot with several regression lines, color=color is not set at the ggplot layer
p <- ggplot(dsmall, aes(x=carat, y=price, group=color)) + geom_point(aes(color=color), size=2) + geom_smooth(aes(color=color), method="lm", se=FALSE) # "group" means grouping ht points
p
p <- ggplot(dsmall, aes(carat, price)) + geom_point(aes(color=color), size=2) + geom_smooth(aes(color=color), method="lm", se=FALSE)
p # same as above! Why?

# the most succinct R script is here ! color=color NOT group=color
p <- ggplot(data=dsmall, mapping=aes(x=carat, y=price, color=color)) + geom_point(size=2) + geom_smooth(method="lm", se=FALSE)
p

# color=color is dropped at the geom_point layer
p <- ggplot(dsmall, aes(carat, price, group=color)) + geom_point(size=4) + geom_smooth(aes(color=color), method="lm", se=FALSE) # All black points but several colorful lines. The difference is that five groups do not mean five colors !!! 七條回歸線, why? group=color
p

# color=color is dropped BOTH at the geom_point and geom_smooth layers
p <- ggplot(dsmall, aes(carat, price, group=color)) + geom_point(size=2) + geom_smooth(method="lm", se=FALSE) # All black points and several lines. The difference is that five groups do not mean five colors !!!
p

p <- ggplot(data=diamonds, aes(carat, price, colour = cut))
#base layer only
p <- p + layer(geom = "point") # same as p <- p + geom_point()
# 舊版不能如此寫
p <- p + geom_point()
p

### ggplot for jitter, 軸加上晃動
p <- ggplot(dsmall, aes(color, price/carat)) + geom_point()
p <- ggplot(dsmall, aes(color, price/carat)) + geom_jitter(alpha=I(1/2), aes(color=color)) # For large datasets with overplotting the alpha aesthetic will make the points more transparent # position_jitter(width = NULL, height = NULL): Defaults to 40% of the resolution of the data. alpha: 點的透明度
# Please check geom_point(alpha=I(1/2), aes(color=color))
p

### ggplot2 line graph
p <- ggplot(iris, aes(x=Petal.Length, y=Petal.Width, color=Species)) + geom_line() # 有沒有分組? yes, 有無不同顏色? yes
p

p <- ggplot(iris, aes(Petal.Length, Petal.Width, group=Species)) + geom_line() # 有沒有分組? yes, 有無不同顏色? no
p # no color=Species, all black lines

### ggplot2 line graph with facets (分面)
p <- ggplot(iris, aes(Sepal.Length, Sepal.Width)) + geom_line(aes(color=Species), size=1) + facet_wrap(~Species, ncol=1) # size=1 makes lines thicker
p

p <- ggplot(iris, aes(Sepal.Length, Sepal.Width)) + geom_line(aes(color=Species), size=1) + facet_wrap(~Species) # ncol=1 is dropped, one row three columns
p

### data preprocessing for line graphs
y <- matrix(rnorm(500), nrow=100, ncol=5, dimnames=list(paste("g", 1:100, sep=""), paste("Sample", 1:5, sep="")))
y <- data.frame(Position=1:length(y[,1]), y) # for using ggplot(...)
y[1:4, ] # A wide format

library(reshape2)
df <- melt(y, id.vars='Position', variable.name='Samples', value.name='Values') # Melt in into a long format
head(df)
### ggplot for line graphs
p <- ggplot(data=df, aes(x=Position, y=Values)) + geom_line(aes(color=Samples)) + facet_wrap(~Samples, ncol=1)
p

### Practice 4: ggplot2 Line Graphs ####
# 套件{datasets}中有一牙齒生長的資料集ToothGrowth，請先進行資料理解，接著繪製不同劑量與不同維他命吸收方式下的牙本質細胞的平均長度折線圖。
### Practices 5~9: 題目請參考單張紙本實作練習 ####

# calculate means and standard deviations by Species
iris_mean <- aggregate(iris[,1:4], by=list(Species=iris$Species), FUN=mean) # 3*5 data frame
iris_sd <- aggregate(iris[,1:4], by=list(Species=iris$Species), FUN=sd) # 3*5 data frame
# transform the iris data into a ggplot2-friendly format
str(iris_mean)
library(reshape2)
df_mean <- melt(data=iris_mean, id.vars='Species', variable.name='Features', value.name='Values')
temp_mean <- dcast(data=df_mean, Species ~ Features, value.var="Values") # temp_mean has same structure as iris_mean

df_sd <- melt(iris_sd, id.vars='Species', variable.name='Features', value.name='Values')
temp_sd <- dcast(df_sd, Species ~ Features, value.var="Values") # temp_sd has same structure as iris_sd

### ggplot2 bar plot
head(df_mean)
p <- ggplot(data=df_mean, aes(x=Features, y=Values, fill = Species)) + geom_bar(stat='identity', position="dodge") # pay attention to fill and 丟summarized好的表，故設定為identity
p # stat='identity', position="dodge"

data(iris)
str(iris)
ggplot(data=iris, aes(x=Species)) + geom_bar()
ggplot(data=iris, aes(x=Species)) + geom_bar(stat='count')
# 丟iris$Species，故需count

### ggplot sideway bar plot
p <- ggplot(data=df_mean, mapping=aes(x=Features, y=Values, fill = Species)) + geom_bar(stat='identity', position="dodge") + coord_flip() + theme(axis.text.y=element_text(angle=0, hjust=1)) # y axis tick labels, angle: angle (in [0,360]), hjust: horizontal justification (in [0, 1])
p

p <- ggplot(df_mean, aes(Features, Values, fill = Species)) + geom_bar(stat='identity', position="dodge") + coord_flip() + theme(axis.text.y=element_text(angle=45, hjust=0.5)) # y axis tick labels, angle: angle (in [0,360]), hjust: horizontal justification (in [0, 1])
p

### ggplot2 bar plot with facets (AGAIN!) 分面圖，上面是疊加圖
p <- ggplot(df_mean, aes(Features, Values)) + geom_bar(stat='identity', aes(fill = Species)) + facet_wrap(~Species, ncol=1)
p

p <- ggplot(df_sd, aes(Features, Values)) + geom_bar(stat='identity', aes(fill = Species)) + facet_wrap(~Species, ncol=1) # for df_sd
p

### an example about ggplot: barplot with missing levels
BOD # Time 6 is missing !
?BOD # The Biochemical Oxygen Demand versus time in an evaluation of water quality.
str(BOD)

ggplot(BOD, aes(x=Time, y=demand)) + geom_bar(stat="identity") # there is not a level 6 !

ggplot(BOD, aes(x=factor(Time), y=demand)) + geom_bar(stat="identity") # use factor(Time) as the x-axis

##### an example about variable reordering (doing in {ggplot2})
library(gcookbook)
rank(uspopchange$Change)
(upc <- subset(uspopchange, rank(Change)>40))
uspopchange$Change
ggplot(upc, aes(x=Abb, y=Change, fill=Region)) + geom_bar(stat="identity")
# mapping occurs within aes(), setting occurs outside of aes()
ggplot(upc, aes(x=reorder(Abb, Change), y=Change, fill=Region)) + geom_bar(stat="identity", color="black") + xlab("State") # Is it better for the visualization effect? Sure.
################################################

### ggplot for bar plot with error bar, geom_errorbar()
# define standard deviation limits
limits <- aes(ymax = df_mean[,3] + df_sd[,3], ymin = df_mean[,3] - df_sd[,3]) # mean + or - sd, aes 美學的映射

p <- ggplot(df_mean, aes(Features, Values, fill=Species)) + geom_bar(stat='identity', position="dodge") + geom_errorbar(limits, position="dodge") ##### map variables to the values for ymin and ymax
p

### ggplot2 for bar plot changing color
library(RColorBrewer)
p <- ggplot(df_mean, aes(Features, Values, fill=Species, color=Species)) + geom_bar(stat='identity', position="dodge") + geom_errorbar(limits, position="dodge") + scale_fill_brewer(palette="Blues") + scale_color_brewer(palette="Greys") # brewer
p

### ggplot2 for bar plot using standard color
p <- ggplot(df_mean, aes(Features, Values, fill=Species, color=Species)) + geom_bar(stat='identity', position="dodge") + geom_errorbar(limits, position="dodge") + scale_fill_manual(values=c("red","green3","blue")) + scale_color_manual(values=c("dark red","dark green","dark blue")) # manual
p

### ggplot: boxplot
p <- ggplot(dsmall, aes(color, price/carat, fill=color)) + geom_boxplot()
p

p <- ggplot(dsmall, aes(color, price/carat, fill=color)) + geom_boxplot() + guides(fill=FALSE)
p

#### ggplot: boxplot with jitter points
data(mtcars)
str(mtcars)
mtcars$cylinder <- as.factor(mtcars$cyl)

# by qplot
qplot(cylinder, mpg, data=mtcars, geom=c("boxplot", "jitter"), fill=cylinder, main="Box plots with superimposed data points", xlab="Number of Cylinders", ylab="Miles per Gallon")

ggplot(data=mtcars, mapping=aes(x=cylinder, y=mpg, fill=cylinder)) + geom_boxplot() + geom_jitter() + ggtitle("Box plots with superimposed data points") + xlab("Number of Cylinders") + ylab("Miles per Gallon")

ggplot(data=mtcars, mapping=aes(x=cylinder, y=mpg, fill=cylinder)) + geom_boxplot() + geom_point() + ggtitle("Box plots with superimposed data points") + xlab("Number of Cylinders") + ylab("Miles per Gallon")

### ggplot: stacked density plot
p <- ggplot(dsmall, aes(x=carat)) + geom_density(aes(color=color))
p
### ggplot: density area plot
p <- ggplot(dsmall, aes(carat)) + geom_density(aes(fill=color)) # what is the visualization effect? Difference between color and fill.
p
### ggplot for histogram with density curve
p <- ggplot(iris, aes(x=Sepal.Width)) + geom_histogram(aes(y=..density.., fill=..count..), binwidth=0.2) + geom_density()
p # ..density..(relative frequency) ..count..(filled color by absolute frequency)

### Practice 5: Histogram
# 請以ggplot2繪圖套件，繪製iris資料集中的Petal.Width變數的直方圖(組數為10)

### ggplot for pie chart
df <- data.frame(variable=rep(c("cat","mouse","dog","bird","fly")), value=c(1,3,3,4,2))
p <- ggplot(df, aes(x='', y=value, fill=variable)) + geom_bar(stat='identity', width=1) + coord_polar("y", start=pi/3) + ggtitle("Pie Chart") # coord_polar 極座標
p
### ggplot: arranging graphs on one page
library(grid) # mostly for developer
a <- ggplot(dsmall, aes(color, price/carat)) + geom_jitter(size=4, alpha=I(1/1.5), aes(color=color))
b <- ggplot(dsmall, aes(color, price/carat, color=color)) + geom_boxplot()
c <- ggplot(dsmall, aes(color, price/carat, fill=color)) + geom_boxplot() + theme(legend.position = "none")
grid.newpage()
pushViewport(viewport(layout = grid.layout(2,2)))
print(a, vp = viewport(layout.pos.row=1, layout.pos.col=1:2))
print(b, vp = viewport(layout.pos.row=2, layout.pos.col=1))
print(c, vp = viewport(layout.pos.row=2, layout.pos.col=2, width=0.3, height=0.3, x=0.8, y=0.8))

print(a)
print(b, vp=viewport(width=0.3, height=0.3, x=0.8, y=0.8))

### ggplot for normal density
p <- ggplot(data.frame(x=c(-3,3)), aes(x=x)) # Base layer (data 指向單欄資料框)
p + stat_function(fun=dnorm) # dnorm: density function for normal

### ggplot for t density
p + stat_function(fun=dt, args=list(df=2))

### ggplot for user-defined function 使用者自訂的函數
myfun <- function(xvar) {
  1/(1 + exp(-xvar + 10)) # Sigmoid function, machine learning often uses
}
ggplot(data.frame(x=c(0,20)), aes(x=x)) + stat_function(fun=myfun)

### ggplot for function curve with a shaded region
dnorm_limit <- function(x) {
  y <- dnorm(x)
  y[x < 0 | x > 2] <- NA
  return(y)
}
p <- ggplot(data=data.frame(x=c(-3,3)), aes(x=x))
p + stat_function(fun=dnorm_limit, geom='area', fill="blue", alpha=0.2) + stat_function(fun=dnorm) # fun=dnorm_limit, geom='area', fill='blue'

### ggplot for function curve (any function) with a shaded region, similar to above 有彈性的寫法
limitRange <- function(fun, min, max) {
  function(x) {
    y <- fun(x)
    y[x < min | x > max] <- NA
    return(y)
  }
}
dlimit <- limitRange(dnorm, 0, 2)
dlimit
dlimit(-2:4)
p + stat_function(fun = dnorm) + stat_function(fun = limitRange(dnorm, 0, 2), geom='area', fill='blue', alpha=0.2) #圖層可以顚倒

### ggplot for ecdf
library(gcookbook)
str(heightweight)
?heightweight
head(heightweight)
ggplot(heightweight, aes(x=heightIn)) + stat_ecdf()
#ecdf: Empirical Cumulative Density Function
### ggplot for ecdf
ggplot(heightweight, aes(x=ageYear)) + stat_ecdf()

### Case 2: Exploratory Data Analysis by ggplot2 ####
custdata <- read.table(file.choose(), header=T, sep='\t') # find custdata.tsv
summary(custdata)
#####
summary(custdata$state.of.res)
sort(table(custdata$state.of.res), decreasing=TRUE)
which(is.na(custdata$housing.type))
which(is.na(custdata$recent.move))
which(is.na(custdata$num.vehicles))
identical(which(is.na(custdata$housing.type)),which(is.na(custdata$recent.move)))# test 遺缺是否同一筆
identical(which(is.na(custdata$recent.move)),which(is.na(custdata$num.vehicles)))
#####

summary(custdata[,c("is.employed", "housing.type", "recent.move", "num.vehicles")])
summary(custdata$income)
summary(custdata$age)
summary(custdata$income)
Income = custdata$income/1000
summary(Income)
library(ggplot2)
ggplot(data=custdata) + geom_histogram(aes(x=age),
                                       binwidth=5, fill="gray")
diff(range(custdata$age))/30 # 4.88934, range/30　是合理組距
ggplot(data=custdata) +
  geom_density(aes(x=age)) #直方圖的連續版本
library(scales)　
ggplot(custdata) + geom_density(aes(x=income)) +
  scale_x_continuous(labels=dollar) # right-skewed
ggplot(custdata) + geom_density(aes(x=income)) +
  scale_x_log10(breaks=c(100,1000,10000,100000), labels=dollar) +
  annotation_logticks(sides="bt") # bottom and top
ggplot(custdata) + geom_bar(aes(x=marital.stat), fill="gray")
ggplot(custdata) +
  geom_bar(aes(x=state.of.res), fill="gray") +
  coord_flip() +
  theme(axis.text.y=element_text(size=rel(0.8)))
statesums <- table(custdata$state.of.res)   # aggregates the data by state of residence -- exactly the information that barchart plots.
statef <- as.data.frame(statesums) 	# Convert the table object to a data frame using as.data.frame(). The default column names are "Var1" and "Freq".
colnames(statef)<-c("state.of.res", "count") 	# Rename the columns for readability.
summary(statef)  	# Notice that the default ordering for the state.of.res variable is alphabetical.
statef <- transform(statef,
                    state.of.res=reorder(state.of.res, count)) # Use the reorder() function to set the state.of.res variable to be count-ordered. Use the transform() function.
#   to apply the transformation to the statef data frame.
summary(statef) # The state.of.res variable is now count ordered.
ggplot(statef)+ geom_bar(aes(x=state.of.res,y=count), stat="identity",	# ***y=count and*** use stat="identity" to plot the data exactly as given.
                         fill="gray") +
  coord_flip() +                                       	# Flip the axes and reduce the size of the label text as before.
  theme(axis.text.y=element_text(size=rel(0.8)))
custdata2 <- subset(custdata, (custdata$age > 0 & custdata$age < 100 & custdata$income > 0))
cor(custdata2$age, custdata2$income)
ggplot(custdata2, aes(x=age, y=income)) + geom_point() + ylim(0, 200000)
ggplot(custdata2, aes(x=age, y=income)) + geom_point() + stat_smooth(method="lm") + ylim(0, 200000)
ggplot(custdata2, aes(x=age, y=income)) +
  geom_point() + geom_smooth() +
  ylim(0, 200000)
ggplot(custdata2, aes(x=age, y=as.numeric(health.ins))) +
  geom_point(position=position_jitter(w=0.05, h=0.05)) +
  geom_smooth()
library(hexbin)
ggplot(custdata2, aes(x=age, y=income)) +
  geom_hex(binwidth=c(5, 10000)) +
  geom_smooth(color="white", se=F) +
  ylim(0,200000)
ggplot(custdata) + geom_bar(aes(x=marital.stat, fill=health.ins)) # x 與 fill 分別對到兩個不同類別變數
ggplot(custdata) + geom_bar(aes(x=marital.stat, fill=health.ins), position="dodge")
ggplot(custdata, aes(x=marital.stat)) +
  geom_bar(aes(fill=health.ins), position="fill") +
  geom_point(aes(y=-0.05), size=0.75, alpha=0.3,
             position=position_jitter(h=0.01))
ggplot(custdata2) +
  geom_bar(aes(x=housing.type, fill=marital.stat ),
           position="dodge") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggplot(custdata2) +
  geom_bar(aes(x=marital.stat), position="dodge",
           fill="darkgray") +
  facet_wrap(~housing.type, scales="fixed") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) # try scales="fixed"

### Part 4 ggmap ####
# David Kahle and Hadley Wickham, ggmap: Spatial Visualization with ggplot2, The R Journal, 2013, Vol.5, No.1, pp.144--161, http://journal.r-project.org/archive/2013-1/kahle-wickham.pdf.
library(ggmap)
baylor <- "baylor university"
qmap(baylor, zoom = 14) # qmap is a wrapper for ggmap and get_map.
?qmap
qmap(baylor, zoom = 14, source = "osm")

##### Step 1: Downloading the map raster
### Define location: 3 ways
myLocation <- "University of Washington"
myLocation <- c(lon = -95.3632715, lat = 29.7632836)
myLocation <- c(-130, 30, -105, 50)
geocode("University of Washington")

### Define map source, type, and color
myMap <- get_map(location=myLocation, source="stamen", maptype="watercolor", crop=FALSE) # step 1, zoom="auto"
ggmap(myMap) # step 2

### Fine tune the scale of the map using zoom
myMap <- get_map(location=myLocation, source="stamen", maptype="watercolor", crop=FALSE, zoom=6) # map zoom (finer), an integer from 3 (continent) to 21 (building), default value 10 (city). 
ggmap(myMap)

##### Step 2: Plotting the maps and data
### Plot the raster
ggmap(myMap)

### Add points with latitude/longitude coordinated
## Case 1: Violent crimes in Houston
data(crime) # Lightly cleaned Houston crime from January 2010 to August 2010 geocoded with Google Maps
str(crime)

# only violent crimes (robbery, aggravated assault, rape, and murder 搶劫、惡意攻擊、強暴與謀殺)
table(crime$offense)
violent_crimes <- subset(crime, offense != "auto theft" & offense != "theft" & offense != "burglary")
# order violent crimes
violent_crimes$offense <- factor(violent_crimes$offense, levels = c("robbery","aggravated assault", "rape", "murder")) #1->2->3->4
# restrict to downtown
violent_crimes <- subset(violent_crimes, -95.39681 <= lon & lon <= -95.34188 & 29.73631 <= lat & lat <= 29.78400)

houston <- get_map("houston", zoom = 14) # 21 is buildings !
(HoustonMap <- ggmap(houston, extent = "device", legend = "topleft")) # extent: how much of the plot should the map take up?

##### 2d density estimation
library("MASS")
data(geyser)
str(geyser) # 噴發間隔時間，噴發持續時間
m <- ggplot(geyser, aes(x = duration, y = waiting)) +
  geom_point() + xlim(0.5, 6) + ylim(40, 110)
m + geom_density2d()

dens <- kde2d(geyser$duration, geyser$waiting, n = 50, lims = c(0.5, 6, 40, 110)) # kernal density estimation (kde)
densdf <- data.frame(expand.grid(duration = dens$x, waiting = dens$y), z = as.vector(dens$z))
m + geom_contour(aes(z=z), data=densdf)
#####

# 點繪圖
HoustonMap + geom_point(aes(x = lon, y = lat), data = violent_crimes, alpha = .5, color="darkred")
# 點繪圖
HoustonMap + geom_point(aes(x = lon, y = lat, colour = offense, size = offense), data = violent_crimes)

# 2d 密度估計水準圖 NOT shown in slides (Why is bins parameter unknown for the stat_density2d function? (ggmap) http://stackoverflow.com/questions/34410999/why-is-bins-parameter-unknown-for-the-stat-density2d-function-ggmap)
HoustonMap + stat_density2d(aes(x = lon, y = lat, fill = ..level.., alpha = ..level..), size = 2, bins = 4, data = violent_crimes, geom = "polygon")
overlay <- stat_density2d(aes(x = lon, y = lat, fill = ..level.., alpha = ..level..), bins = 4, geom = "polygon", data = violent_crimes)
# library(grid)
# HoustonMap + overlay + inset( grob = ggplotGrob(ggplot() + overlay + theme_inset()), xmin = -95.35836, xmax = Inf, ymin = -Inf, ymax = 29.75062)

##### Visualization, ggplot2 and ggmap (ggplot2 + map)
library(ggmap)
library(mapproj)
map <- get_map(location = 'Taiwan', zoom = 7) # 地圖的位置是透過 location 參數來指定，直接輸入地名即可，而 zoom 則是控制地圖的大小 an integer from 3 (continent) to 21 (building), default value 10 (city). 
ggmap(map)

map <- get_map(location = 'Taiwan', zoom = 7,language = "zh-TW") # language 可以設定地圖上文字標示的語言
ggmap(map)

map <- get_map(location = c(lon = 120.233937, lat = 22.993013),zoom = 10, language = "zh-TW") # 需要畫出比較精確的位置時 location 參數也可以接受經緯度
ggmap(map)


map <- get_map(location = c(lon = 120.233937, lat = 22.993013),zoom = 15, language = "zh-TW", maptype = "roadmap") # maptype 參數可以指定地圖的類型（預設是 terrain）/ satellite/ hybrid/ toner-lite, try zoom = 18
ggmap(map)
ggmap(map, darken = 0.5) # darken 這個參數可以讓地圖變暗（或是變亮）
ggmap(map, darken = c(0.5, "white"))

# 將資料畫在地圖上
# 從政府資料開放平臺上下載紫外線即時監測資料的 csv 檔，接著將資料讀進 R中。
uv <- read.csv("UV_20151116152215.csv")
lon.deg <- sapply((strsplit(as.character(uv$WGS84Lon), ",")), as.numeric)
uv$lon <- lon.deg[1, ] + lon.deg[2, ]/60 + lon.deg[3, ]/3600 # 這裡原始的經緯度資料是以度分秒表示，在使用前要轉換為度數表示。
lat.deg <- sapply((strsplit(as.character(uv$WGS84Lat), ",")), as.numeric)
uv$lat <- lat.deg[1, ] + lat.deg[2, ]/60 + lat.deg[3, ]/3600

map <- get_map(location = 'Taiwan', zoom = 7)
ggmap(map) + geom_point(aes(x = lon, y = lat, size = UVI, color='pink'), data = uv)  # ggmap 負責畫出基本的地圖，然後再使用 geom_point 加上資料點，除了指定經緯度之外，我們還使用紫外線的強度（UVI）來指定圓圈的大小。

### Part 4-1 Geocoding with Tidygeocoder (Run this part in Terminal on Mac) ####
# https://jessecambon.github.io/2019/11/11/tidygeocoder-demo.html
library(dplyr)
library(tidygeocoder)

dc_addresses <- tribble(~name, ~addr,
                         "White House", "1600 Pennsylvania Ave Washington, DC",
                         "National Academy of Sciences", "2101 Constitution Ave NW, Washington, DC 20418",
                         "Department of Justice", "950 Pennsylvania Ave NW, Washington, DC 20530",
                         "Supreme Court", "1 1st St NE, Washington, DC 20543",
                         "Washington Monument", "2 15th St NW, Washington, DC 20024")

coordinates <- dc_addresses %>%
  geocode(addr)

library(OpenStreetMap)
dc_map <- openmap(c(38.905,-77.05), c(38.885,-77.00)) # downloads a street map
dc_map.latlng <- openproj(dc_map) # projection = "+proj=longlat", Projects the open street map onto a latitude and longitude coordinate system so that we can overlay our coordinates

library(ggplot2)
library(ggrepel)
autoplot(dc_map.latlng) +
  theme_minimal() +
  theme(      axis.text.y=element_blank(),
              axis.title=element_blank(),
              axis.text.x=element_blank(),
              plot.margin = unit(c(0, 0, 0, 0), "cm")
  ) +
  geom_point(data=coordinates, aes(x=long, y=lat), color="navy", size=4, alpha=1) +
  geom_label_repel(data=coordinates,
                   aes(label=name,x=long, y=lat),show.legend=F,box.padding=.5,size = 5)

### Part 5 Wordcloud ####
# http://chengjun.github.io/cn/2013/09/topic-modeling-of-song-peom/
library(Rwordseg)
require(rJava)
library(tm) # version 0.5-10
library(slam) # sparse matrix
# library(topicmodels)
library(wordcloud)
#library(igraph)

txt <- read.csv(file.choose(), colClasses="character", header=T) # select songPoem_win.csv

# 宋詞欣賞
unique(txt$Author)
txt$Author[grep("蘇軾",txt$Author)]
txt$Author[grep("辛棄疾",txt$Author)]
txt$Author[grep("李清照",txt$Author)]

txt$Sentence[grep("蘇軾",txt$Author)]

# 中文分詞
segmentCN(txt$Sentence[1])
poem_words <- lapply(1:length(txt$Sentence), function(i) segmentCN(txt$Sentence[i], nature = TRUE)) # nature: Whether to recognise the nature of the words. 參見Rwordseg_Vignette_CN.pdf表1

poem_words[1:5]

class(poem_words) # "list"
length(poem_words) # 20692 elements
# poem_words[[1]]
# txt$Sentence[1]

# 建語料庫
wordcorpus <- Corpus(VectorSource(poem_words)) # from "tm"

# 建文件詞項矩陣
dtm1 <- DocumentTermMatrix(wordcorpus, control = list(wordLengths=c(1, Inf), # to allow long words
                                                      bounds = list(global = c(5,Inf)), # each term appears in at least 5 docs
                                                      removeNumbers = TRUE, weighting = weightTf, encoding = "UTF-8")) # 20692 * 9030

colnames(dtm1)
findFreqTerms(x=dtm1, lowfreq=1000) # 看一下高頻詞(242), lowfreq = 1000, highfreq = Inf
?findFreqTerms

dtm2 <- DocumentTermMatrix(wordcorpus, control = list(wordLengths=c(2, Inf), bounds = list(global = c(5,Inf)), removeNumbers = TRUE, weighting = weightTf,encoding = "UTF-8")) # 20692 * 4960, Inf: infinite

dtm3 <- DocumentTermMatrix(wordcorpus, control = list(wordLengths=c(3, Inf), bounds = list(global = c(5,Inf)), removeNumbers = TRUE, weighting = weightTf, encoding = "UTF-8")) # 20692 * 171

# 詞頻計算
m <- as.matrix(dtm1) # as a regular matrix -> column sum
v <- sort(colSums(m), decreasing=TRUE)
myNames <- names(v) # 9030
d <- data.frame(word=myNames, freq=v) # 建字詞與詞頻資料框
#par(mar = rep(2, 4), family="STKaiti") # 中文字形 mac
pal2 <- brewer.pal(8,"Dark2")
wordcloud(d$word, d$freq, scale=c(5,.2), min.freq=mean(d$freq), max.words=100, random.order=FALSE, rot.per=.15, colors=pal2) # 人、不、春、花...

# 詞頻計算
m <- as.matrix(dtm2)
v <- sort(colSums(m), decreasing=TRUE)
myNames <- names(v)
d <- data.frame(word=myNames, freq=v) # 建字詞與詞頻資料框
par(mar = rep(2, 4), family="STKaiti") # 中文字形
pal2 <- brewer.pal(8,"Dark2")
wordcloud(d$word, d$freq, scale=c(5,.2), min.freq=mean(d$freq), max.words=100, random.order=FALSE, rot.per=.15, colors=pal2) # 東風、何處、人間...

# 詞頻計算
m <- as.matrix(dtm3)
v <- sort(colSums(m), decreasing=TRUE)
myNames <- names(v)
d <- data.frame(word=myNames, freq=v) # 建字詞與詞頻資料框
par(mar = rep(2, 4), family="STKaiti") # 中文字形
pal2 <- brewer.pal(8,"Dark2")
wordcloud(d$word, d$freq, scale=c(5,.2), min.freq=mean(d$freq), max.words=100, random.order=FALSE, rot.per=.15, colors=pal2) # 三十六、海棠花、二十四、歸去來兮...

### Part 5-1 Rmashup.R (TODO) ####

### Part 6 Mosaic, fourfold, heat map... ####
### Case form
library(vcd) # visualizing categorical data
names(Arthritis)
str(Arthritis)
head(Arthritis, 5)
nlevels(Arthritis$Improved)

### Frequency form (LONG FORM)
GSS <- data.frame(
  expand.grid(sex=c("female", "male"),
              party=c("dem", "indep", "rep")),
  count=c(279,165,73,47,225,191))
GSS
names(GSS)
str(GSS)
sum(GSS$count) # total number of obs. (總觀測數)
nrow(GSS) ###### no. of cells or no. of all combinations of all factors (各因子總組合數)

### Table form (WIDE FORM)
str(HairEyeColor) # class 'table'
HairEyeColor
sum(HairEyeColor) # the total number of observations
length(dimnames(HairEyeColor)) ##### it's a list, three-way contingency table
sapply(dimnames(HairEyeColor), length) # What is the class of dimnames's output? Ans. A list. So, use sapply().

### Sieve diagrams
data("HairEyeColor") # Hair-Eye-Sex three-way contigency table

## aggregate over 'sex':
(tab <- margin.table(HairEyeColor, c(2,1))) # Hair-Eye changed to Eye-Hair
library('vcd') # for sieve(...)
## plot expected values:
sieve(tab, sievetype = "expected", shade = TRUE)

## plot observed table:
sieve(tab, shade = TRUE)

### Condensed Mosaic displays for two-way tables
HairEye <- margin.table(HairEyeColor, c(2,1)) # make an Eye-Hair two-way contingency table
HairEye
mosaic(HairEye, main = "Basic Mosaic Display of Hair Eye Color data") # Eye: h, Hair: v
mosaic(~ Hair + Eye, data=HairEye, direction=c('v','h'), main = "Basic Mosaic Display of Hair Eye Color data")

?mosaic

### Enhanced Mosaic displays for two-way tables
(HairEye <- as.table(HairEye[c(1,3,4,2),]))
str(HairEye)
mosaic(~ Hair + Eye, data=HairEye, direction=c('v','h'), shade=T, gp_args = list(lty = c(1, 2)), main = "Enhanced Mosaic Display of Hair Eye Color data")
mosaic(~ Eye + Hair, data=HairEye, direction=c('v','h'), shade=T, gp_args = list(lty = c(1, 2)), main = "Enhanced Mosaic Display of Hair Eye Color data")

### Three-way Mosaic displays
HairEyeColor
str(HairEyeColor)
(HairEyeColor <- as.table(HairEyeColor[,c(1,3,4,2),])) # Brown, Hazel, Blue, Green
dimnames(HairEyeColor)
dimnames(HairEyeColor)$Sex <- c("M","F") # avoid words overlapping
mosaic(~ Hair + Eye + Sex, data=HairEyeColor, direction=c('v','h','v'), shade=T, gp_args = list(lty = c(1, 2)), main = "Three-way Mosaic Display of Hair Eye Color data")

### UCBAdmission
ftable(UCBAdmissions) # flatten the three-way table
dimnames(UCBAdmissions)
# library(vcd)
mosaic(~ Admit + Gender + Dept, data=UCBAdmissions) # 1. more applicants were rejected than admitted, 2. there were more men than women within the admitted group, 3. there approximately the same number of men and women in the rejected group

mosaic(~ Dept + Gender + Admit, data=UCBAdmissions, highlighting="Admit", highlighting_fill=c("lightblue", "pink"), direction=c("v","h","v")) # Admit是最後一個分割變數
# direction的設定方便我們比較各系之男女群組

mosaic(~ Dept + Gender + Admit, data=UCBAdmissions, highlighting="Admit", highlighting_fill=c("lightblue", "pink"), direction=c("v","v","h")) # 容易比較各系/跨系男女入學率

mosaic(~ Dept + Gender + Admit, data=UCBAdmissions, highlighting="Admit", highlighting_fill=c("lightblue", "pink"), direction=c("v","h","h")) # 容易比較跨系男女申請率

### fourfold display
data("UCBAdmissions")
UCBAdmissions
str(UCBAdmissions)
x <- aperm(UCBAdmissions, c(2, 1, 3))
x
str(x)
dimnames(x)[[2]] <- c("Yes", "No")
names(dimnames(x)) <- c("Sex", "Admit?", "Department")
ftable(x)

## Fourfold display of data aggregated over departments, with frequencies standardized to equate the margins for admission and sex.
fourfold(margin.table(x, c(1, 2)))

### ### multiple strata fourfold display
fourfold(x)

### spine (It shows how a categorical response varies with a continuous or categorical predictor) and conditional density plots (A further generalization, smoothing rather than discretizing the explanatory variable)
(spine(Improved ~ Age, data=Arthritis, breaks=3)) # pay your attention to the widths and same height
(spine(Improved ~ Age, data=Arthritis, breaks='Scott')) # there is a small group (70,80]

cdplot(Improved ~ Age, data=Arthritis) # why is there a little some improved between 40 and 50? More some and marked improved after 50 !
with(Arthritis, rug(jitter(Age), col="white", quiet=FALSE)) 
# quiet: logical indicating if there should be a warning about clipped values

### heatmap {stats}
heatmap(as.matrix(mtcars), # object must be a matrix !
        Rowv=NA, 
        Colv=NA, 
        col = heat.colors(256), 
        scale="column", # if the values should be centered and scaled in either the row direction or the column direction, or none
        margins=c(3,8), # margins for column and row names, respectively
        main = "Car characteristics by Model")
?heatmap

heatmap(as.matrix(mtcars), 
        col = rainbow(256), 
        scale="column", 
        margins=c(3,8), 
        main = "Car characteristics by Model")

### Creating a Heat Map for time series data
presidents
str(presidents)
help(presidents)
class(presidents) # class 'ts'
methods(ts) # warning: 'ts' appears not to be a generic

plot(presidents, las = 1, ylab = "Approval rating (%)", main = "presidents data")

time(presidents) # 1945.00, 1945.25, 1945.50, 1945.75, is it interesting?
class(time(presidents)) # class 'ts'
plot(as.vector(time(presidents)), as.vector(presidents), type = "l")

nrow(presidents) # NULL
# create a data frame for president rating
pres_rating <- data.frame(rating=as.numeric(presidents), year=as.numeric(floor(time(presidents))), quarter=as.numeric(cycle(presidents)))
# doing some inside-outs
time(presidents) # same as above! Time-Series: 30 * 4 (1945, 1945.25, 1945.5, 1945.75)
cycle(presidents) # Time-Series: 30 * 4 (1, 2, 3, 4)

head(pres_rating)
dim(pres_rating) # 120 * 3 (rating, year, quarter)

library('ggplot2')
p <- ggplot(data=pres_rating, aes(x=year, y=quarter, fill=rating)) # heat map for rating by year * quarter
p + geom_tile()

p + geom_raster()

p + geom_tile() + scale_x_continuous(breaks=seq(1940, 1976, by=4)) + scale_y_reverse() + scale_fill_gradient2(midpoint=50, mid="grey70", limits=c(0,100)) # smooth gradients among three colors. I like grey70.

### Visualizing time series as calendar heat maps 
library(tseries)
# TW <-get.hist.quote("^TWII","2002-03-08","2009-02-02","Close",compression="w")
stock.data <- get.hist.quote("^TWII","2006-01-01","2010-12-31",compression="d")
class(stock.data) # it's a zoo object
head(stock.data)
time(stock.data) # get the time stamps
library(makeR) # for CalendarHeat
calendarHeat(dates=time(stock.data),
             values=stock.data$Close,
             varname="Yahoo Close")

library(openair)
calendarPlot(mydata)
mydata$sales<-rnorm(length(mydata$nox),mean=1000,sd=1500)
calendarPlot(mydata,pollutant="sales",main="Daily Sales in 2003")

### Another way to create a Correlation Heat Map {corrplot}
mtcars
(mcor <- cor(mtcars))
round(mcor, digits=2) # Print mcor and round to 2 digits
library(corrplot)
corrplot(mcor)
with(mtcars, plot(am, mpg))
with(mtcars, plot(cyl, hp)) # method is defaulted to 'circle'

corrplot(mcor, method="shade", shade.col=NA, tl.col="black", tl.srt=45) # Method "pie" and "shade" came from Michael Friendly's job, tl.col: text color, tl.srt: text label string rotation in degrees

corrplot(mcor, method="pie", tl.col="black", tl.srt=45)

col <- colorRampPalette(c('#BB4444', '#EE9988', '#FFFFFF', '#77AADD', '#4477AA'))
corrplot(mcor, method="shade", shade.col=NA, tl.col="black", tl.srt=45, col=col(200), addCoef.col="black", order='AOE') # addCoef.col=Color of coefficients added on the graph, order={"AOE"|"FPC"|"hclust"}: the ordering method of the correlation matrix
?corrplot

##### Correlograms are a relatively recent tool for visualizing the data in correlation matices
options(digits=2) # 為了有效數字
library(corrgram)
corrgram(mtcars, order=T, lower.panel=panel.shade, upper.panel=panel.pie, text.panel=panel.txt, main='Correlogram of mtcars intercorrelations')

### Creating a Choropleth Map
USArrests
crimes <- data.frame(state=tolower(rownames(USArrests)), USArrests)
crimes # state, Murder, Assault, UrbanPop, Rape
require('maps') # library(maps)
require('ggplot2')
# map('state', fill = TRUE, col = palette())
?map_data
states_map <- map_data("state") # United States State Boundaries Map
head(states_map) # long, lat, group, order, region, subregion
crime_map <- merge(states_map, crimes, by.x="region", by.y="state") # merge the data set together
head(crime_map) # after merging, the order has changed, which would lead to polygons drawn in the incorrect order. So, we need to sort the data
?merge
library('plyr') # for arrange() function
crime_map <- arrange(crime_map, group, order) # sort by group, then order
head(crime_map)

library("mapproj")
ggplot(crime_map, aes(x=long,y=lat,group=group,fill=Assault)) + geom_polygon(colour='black') + coord_map('polyconic')

ggplot(crimes, aes(map_id=state,fill=Assault)) + geom_map(map=states_map, colour='black') + scale_fill_gradient2(low='#559999',mid='grey90',high='#BB650B', midpoint=median(crimes$Assault)) + expand_limits(x=states_map$long,y=states_map$lat) + coord_map('polyconic') # how the values diverge from some middle value, please use scale_fill_gradient2()

### Part 7 Interactive Graphics with the {iplot} package (多圖互動) ####
# NOT for Mac OS X
library(iplots)
data(mtcars)
attach(mtcars)
names(mtcars)
cylinders <- factor(cyl)
gears <- factor(gear)
transmission <- factor(am)
ihist(mpg)
ibar(gears)
iplot(mpg, wt)
ibox(mtcars[c("mpg", "wt", "qsec", "disp", "hp")])
ipcp(mtcars[c("mpg", "wt", "qsec", "disp", "hp")])
imosaic(transmission, cylinders)
detach(mtcars)

### Supplement 9: interactive barplots by googleVis (單圖互動) ####
# install.packages("googleVis")
library(googleVis)
stock = read.csv("spq.csv", header = TRUE)
barpt = gvisBarChart(stock, xvar = "Date", yvar = c("Returns"), options = list(orientation = "horizontal", width = 1400, height = 500, title = "Microsoft returns over 2 year period", legend = "none", hAxis = "{title :'Time Period',titleTextStyle :{color:'red'}}", vAxis = "{title : 'Returns(%)', ticks : [-12,-6,0,6, 12],titleTextStyle :{color: 'red'}}", bar = "{groupWidth: '100%'}"))
plot(barpt)

### barplot and Table
# library(googleVis)
# stock = read.csv("spq.csv", header = TRUE)
barpt = gvisBarChart(stock, xvar = "Date", yvar = c("Returns"), options = list(orientation = "horizontal", width = 1400, height = 500, title = "Microsoft returns over 2 year period", legend = "none", hAxis = "{title :'Time Period',titleTextStyle :{color:'red'}}", vAxis = "{title : 'Returns(%)', ticks : [-12,-6,0,6, 12],titleTextStyle :{color: 'red'}}", bar = "{groupWidth: '100%'}"))

table <- gvisTable(stock, options=list(page='enable', height='automatic', width='automatic'),formats = list(Returns =' #.##'))
comb = gvisMerge(table,barpt, horizontal = TRUE)
plot(comb)

### Supplement 10: interactive barplots by plotly (單圖互動) ####
# install.packages("plotly")
library(plotly)
data(economics)
plot_ly(economics, x = economics$date, y = economics$unemploy / economics$pop)

# A plotly visualization is composed of one (or more) trace(s), and every trace has a type (the default type is ‘scatter’). The arguments/properties that a trace will respect (documented here) depend on it’s type. A scatter trace respects mode, which can be any combination of “lines”, “markers”, “text” joined with a “+”:
plot_ly(economics, x = economics$date, y = economics$unemploy/economics$pop, type = "scatter", mode = "markers+lines")

# You can manually add a trace to an existing plot with add_trace(). 
m <- loess(economics$unemploy / economics$pop ~ as.numeric(economics$date), data = economics)
p <- plot_ly(economics, x = economics$date, y = economics$unemploy / economics$pop, name = "raw") 
add_trace(p, x = economics$date, y = fitted(m), name = "loess")

# plotly was designed with a pure, predictable, and pipeable interface in mind, so you can also use the %>% operator to create a visualization pipeline:
economics %>%
  plot_ly(x = economics$date, y = economics$unemploy / economics$pop) %>% 
  add_trace(x = economics$date, y = fitted(m)) %>%
  layout(showlegend = F) # 同上，但沒有legend

# Furthermore, plot_ly(), add_trace(), and layout(), all accept a data frame as their first argument and output a data frame. As a result, we can inter-weave data manipulations and visual mappings in a single pipeline.(以pipeline語法結合資料操弄與視覺化呈現)
economics %>%
  transform(rate = economics$unemploy / economics$pop) %>%
  plot_ly(x = economics$date, y = economics$rate) %>% 
  subset(economics$rate == max(economics$rate)) %>%
  layout(
    showlegend = F, 
    annotations = list(x = economics$date, y = economics$rate, text = "Peak", showarrow = T) # Peak下方的箭頭
  )

s <- plot_ly(z = volcano, type = "surface")

### Supplement 11: Exploratory Analysis of Financial Data ####
## A. Plotting Financial Time Series
# return series plots
library(fPortfolio)
class(LPP2005.RET) # 'timeSeries'
colnames(LPP2005.RET) # Bonds: SBI & LMI, Equities: SPI, MPI, ALT
head(LPP2005.RET)

str(LPP2005.RET)
showClass('timeSeries')

plot(LPP2005.RET, main="LPP Pension Fund", col="steelblue")

# index/return series plot
head(SWX,3)
head(SWX.RET,3)
plot(SWX[,6:4], plot.type="single", col=2:4, xlab="Date", ylab="LP Index Family")
title(main="LP25-LP40-LP60") # add title
hgrid() # dashed horizontal lines

# scatter plot
SBI.RET <- 100*SWX.RET[,"SBI"] # univarite time series
SPI.RET <- 100*SWX.RET[,"SPI"] # univarite time series
plot(SBI.RET, SPI.RET, xlab="SBI", ylab="SPI",pch=19,cex=0.4,col="brown") # some kind of negative linear relation
grid()

# function seriesPlot, returnPlot, cumulatedPlot
SPI <- SWX[,"SPI"] # SPI index series
SPI.RET <- SWX.RET[,"SPI"] # SPI return series
seriesPlot(SPI)
seriesPlot(SPI, type="h") # "h": histogram
returnPlot(SPI) # it's stationary
cumulatedPlot(SPI.RET, index=100) # going back, right? index*exp(colCumsums(x))

op <- par(mfcol = c(3, 2))
seriesPlot(SWX)
par(op)

args(seriesPlot)
args(returnPlot)
args(cumulatedPlot) # index=100


seriesPlot(SPI, labels = FALSE, type = "h", col = "brown", title = FALSE, grid = FALSE, rug = FALSE) # type='h'
lines(SPI, col = "orange") # low level plotting, adding SPI lines
title(main = "Swiss Performance Index")
hgrid()
box_()
copyright()
mtext("SPI", side = 3, line = -2, adj = 1.02, font = 2)

# drawdown series using {fBasics}
drawdownPlot(returns(SPI, method = "discrete")) # compute the returns first

########## Supplement about drawdowns
## Use Swiss Pension Fund Data Set of Returns - 
head(LPP2005REC)
SPI <- LPP2005REC[,'SPI']
head(SPI)
## Plot Drawdowns
dd = drawdowns(LPP2005REC[,"SPI"], main='Drawdowns')
str(dd)
plot(dd) # by {graphics}
# for multivariatr series
dd = drawdowns(LPP2005REC[, 1:6], main = "Drawdowns")
str(dd)
plot(dd)
## Compute Drawdowns Statistics - 
ddStats <- drawdownsStats(SPI)
str(ddStats) # class 'data.frame'
ddStats
sum(ddStats$Length) # 329, NOT 377
## Note, Only Univariate Series are allowd -
ddStats <- try(drawdownsStats(LPP2005REC))
class(ddStats) # class "try-error"
str(ddStats)
##########

# lowess series
SPI <- SWX[,"SPI"] # SPI index series
# x <- SPI # it's meaningless!
class(SPI) # class "timeSeries"
str(SPI)
class(series(SPI)) # " class "matrix"
series(SPI) <- lowess(SPI = 1:nrow(SPI), y = as.vector(SPI), f = 0.1)$y # f: the smoother span. This gives the proportion of points in the plot which influence the smooth at each value. Larger values give more smoothness.
seriesPlot(SPI, rug = FALSE, col = "red", ylim = c(2500, 8000), lwd = 2) # smooth lowess curve first, min(SPI)=2603.37, max(SPI)=7655.55
lines(SPI) # imposing the SPI lines

# axis label customization
plot(SPI, xlab = "", col = "steelblue")
plot(SPI, format = "%b-%Y", xlab = "", col = "steelblue")

# box plot
args(boxPlot) # {fBasics}
boxPlot(returns(SWX))

# box percentile plot
args(boxPercentilePlot)
boxPercentilePlot(returns(SWX))

# histogram + density plot
histPlot(SPI.RET) # rug有助於了解尾部密度
?histPlot

# density plot
args(densityPlot)
densityPlot(SPI.RET)

# qq-plot (quantile-quantile plot)
set.seed(1953)
x <- rnorm(250)
qqnormPlot(x) # {fBasics}

y <- rnig(250)
qqnigPlot(y)

z <- rght(250)
qqghtPlot(z)

## B. Modeling Asset Retuens
# testing asset returns for Normality
args(assetsTest)

shapiroTest <- assetsTest(LPP2005.RET[, 1:3], method = "shapiro") # {fAssets}
shapiroTest <- mvshapiroTest(LPP2005.RET[, 1:3]) # {fAssets}

print(shapiroTest)

assetsTest(LPP2005.RET[, 1:3], method = "energy")

## C. Selecting Similar or Dissimilar Assets
# grouping similar assets
args(assetsSelect)

# grouping asset returns by hierarchical clustering
lppData <- LPP2005.RET
hclustComplete <- assetsSelect(lppData, method = "hclust")
hclustComplete
summary(hclustComplete)

plot(hclustComplete, hang=-1, xlab = "LPP2005 Assets")
mtext("Distance Metric: Euclidean", side = 3)
rect.hclust(hclustComplete, k=2) # separate them into two groups

# grouping asset returns by k-means clustering
kmeans <- assetsSelect(lppData, method = "kmeans", control <- c(centers = 2, algorithm = "Hartigan-Wong"))
sort(kmeans$cluster)


## D. Comparing Multivariate Return and Risk Statistics
# star and segment plots
args(stars)

# library(fPortfolio)
args(assetsStarsPlot)

args(assetsBasicStatsPlot)

lppData <- LPP2005.RET
assetsBasicStatsPlot(lppData[, -8], title = "", description = "")
assetsMomentsPlot(lppData[, -8])
assetsBoxStatsPlot(lppData[, -8], title = "", description = "")


## E. Pairwise Dependencies of Assets
# pairwise scatter plot
args(assetsPairsPlot)
colnames(LPP2005.RET) # SBI, SPI, SII, LMI, MPI, ALT, LPP25, LPP40, LPP60
Assets <- assetsArrange(LPP2005.RET[, 1:6], method = "hclust")
Assets # SII, SBI, LMI, SPI, MPI, ALT
LPP2005HC <- 100 * LPP2005.RET[, Assets] # attention, Assets

##########
hclustComplete <- assetsSelect(LPP2005.RET[, 1:6], method = "hclust")
plot(hclustComplete, hang=-1, xlab = "LPP2005[,1:6] Assets")
mtext("Distance Metric: Euclidean", side = 3)
rect.hclust(hclustComplete, k=2)
##########

assetsPairsPlot(LPP2005HC, pch = 19, cex = 0.5, col = "royalblue4")

histPanel <- function(x, ...) {
  usr <- par("usr")
  on.exit(par(usr))
  par(usr = c(usr[1:2], 0, 1.5))
  h <- hist(x, plot = FALSE)
  breaks <- h$breaks
  nB <- length(breaks)
  y <- h$counts
  y <- y/max(y)
  rect(breaks[-nB], 0, breaks[-1], y, ...)}

assetsPairsPlot(LPP2005HC, diag.panel = histPanel, pch = 19, cex = 0.5, col = "red4", tick = 0, col.axis = "white")

# pairwise correlation 
args(assetsCorgramPlot)
assetsCorgramPlot(LPP2005HC, pch = 19, cex = 0.5) # method="pie"

assetsCorgramPlot(LPP2005HC, method = "shade", pch = 19, cex = 0.5)


args(assetsCorTestPlot)
assetsCorTestPlot(LPP2005HC)


args(assetsCorImagePlot)
assetsCorImagePlot(LPP2005HC)
# set.seed(1953)
# index <- sample(1:ncol(LPP2005HC))
# assetsCorImagePlot(LPP2005HC[, index])

library(fMultivar)
hexHist <- hexBinning(SWX.RET[, c("SBI", "SPI")], bin = 20)
plot(hexHist, xlab = "SBI", ylab = "SPI", col = rev(greyPalette(20)))
title(main = "Bivariate Histogram Plot")

### Supplement 12: Earthquakes Visualization ####
## This part of R script is adapted from http://xccds1977.blogspot.tw/2012/06/ggmap.html
library(ggmap) # ggplot2 + map
library(animation)
library(XML)

webpage <- 'http://webcache.googleusercontent.com/search?q=cache:WDgPChHp888J:data.earthquake.cn/moreDetails.do+&cd=1&hl=zh-TW&ct=clnk&gl=tw'

tables <- readHTMLTable(webpage, stringsAsFactors = FALSE)
sapply(tables, nrow)
head(tables)

raw <- tables[[6]]
head(raw)
names(raw)

data <- raw[,c(1,3,4)]
head(data)
names(data) <- c('date','lat','lon')
str(data) # all variables are in character types
data$lat <- as.numeric(data$lat)
data$lon <- as.numeric(data$lon)
data$date <- as.Date(data$date,  "%Y-%m-%d")

# Get map data by {ggmap} and plot data on the map
ggmap(get_googlemap(center = 'china', zoom=3, maptype='terrain'),extent='device') +
  geom_point(data=data,aes(x=lon,y=lat),colour = 'red',alpha=0.7) +
  stat_density2d(aes(x=lon,y=lat,fill=..level..,alpha=..level..),size=2,bins=4,data=data,geom='polygon') +
  theme(legend.position = "none") # opts has to be changed to theme

# Prepare a plotting function
plotfunc <- function(x) {
  df <- subset(data,date <= x)
  df$lat <- as.numeric(df$lat)
  df$lon <- as.numeric(df$lon)
  p <- ggmap(get_googlemap(center = 'china', zoom=4,maptype='terrain'),,extent='device')+
    geom_point(data=df,aes(x=lon,y=lat),colour = 'red',alpha=0.7)
}

# Get the dates of earthquakes
time <- sort(unique(data$date))

# Generate a GIF file
saveGIF(for(i in time) print(plotfunc(i))) # saveGIF{animation}: Convert images to a single animation file (typically GIF) using ImageMagick or GraphicsMagick

# saveMovie(for(i in time) print(plotfunc(i)))

### Supplement 13: Credit risk management by C5.0 and Visualizing Model Performance ####
# Dataset: germancredit.csv
credit <- read.csv(file.choose()) # select germancredit.csv
str(credit)
credit$Default <- factor(credit$Default, labels=c("No", "Yes"))
summary(credit)

## re-level variables
levels(credit$checkingstatus1) = c("< 0 DM","0-200 DM","> 200 DM","no account") #DM 德國馬克
table(credit$checkingstatus1)

levels(credit$history) = c("good","good","poor","poor","terrible")
table(credit$history)

levels(credit$purpose) <- c("newcar","usedcar",rep("goods/repair",4),"edu",NA,"edu","biz","biz")
table(credit$purpose, useNA="ifany")

levels(credit$savings) = c("< 100 DM","100-500 DM","500-1000 DM","> 1000 DM", "unknown/no account")
table(credit$savings)

levels(credit$employ) = c("unemployed","< 1 year","1-4 years","4-7 years", "> 7 years")
table(credit$employ)

levels(credit$status) = c("M/Div/Sep","F/Div/Sep/Mar","M/Single","M/Mar/Wid")
table(credit$status)

levels(credit$others) = c("none","co-applicant","guarantor")
table(credit$others)

levels(credit$property) = c("real estate","life insurance","car or other", "unknown/no property")
table(credit$property)

levels(credit$otherplans) = c("bank","stores","none")
table(credit$otherplans)

credit$rent <- factor(credit$housing=="A151")
table(credit$rent)
credit$housing <- NULL

levels(credit$job) = c("unemployed","unskilled","skilled", "mgt/self-employed")
table(credit$job)

levels(credit$tele) = c("none","yes")
table(credit$tele)

table(credit$foreign)
levels(credit$foreign) = c("foreign","german")
table(credit$foreign)

head(credit)
summary(credit) # check out the data

prop.table(table(credit$checkingstatus1, credit$Default), 1)
prop.table(table(credit$savings, credit$Default), 1)

summary(credit$duration)
summary(credit$amount)
table(credit$Default) # one-way contingency table

set.seed(1234)
idx <- sample(1:1000, 900)
credit_train <- credit[idx,]
credit_test <- credit[-idx,]
prop.table(table(credit_train$Default)) # 不違約, 違約
prop.table(table(credit_test$Default))

library(C50) # a newer decision tree package C5.0 decision tree
credit_model <- C5.0(credit_train[-1], credit_train$Default) # build the default decision tree by 屬性矩陣與類別標籤向量(以,分隔)
# credit_model <- C5.0(Default ~ ., data=credit_train) # build the default decision tree by 模型公式語法

credit_model # display simple facts about the tree
summary(credit_model) # display detailed information about the tree (tree size = 64, 13.3%), 過度配適

# create a factor vector of predictions on test data
(credit_pred <- predict(credit_model, credit_test))

library(gmodels) # Various R programming tools for model fitting
CrossTable(credit_test$Default, credit_pred, prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE, dnn = c('actual default', 'predicted default')) # 25%

### Compare with another two models - the first one
credit_model1 <- C5.0(credit_train[-1], credit_train$Default, control=C5.0Control(winnow=T, minCases=10)) # build a simpler decision tree, winnow = T, minCases = 10
credit_model1 
summary(credit_model1) # (tree size = 21, 20.0%)
# create a factor vector of predictions on test data
(credit_pred1 <- predict(credit_model1, credit_test))

library(gmodels) # Various R programming tools for model fitting
CrossTable(credit_test$Default, credit_pred1, prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE, dnn = c('actual default', 'predicted default')) # 27%

### Compare with another two models - the second one
credit_model2 <- C5.0(credit_train[-1], credit_train$Default, control=C5.0Control(subset=F, minCases=10)) # build a simpler decision tree, subset = F(不要group), minCases = 10
credit_model2 
summary(credit_model2) # (tree size = 13, 20.9%)
# create a factor vector of predictions on test data
(credit_pred2 <- predict(credit_model2, credit_test))

library(gmodels) # Various R programming tools for model fitting
CrossTable(credit_test$Default, credit_pred2, prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE, dnn = c('actual default', 'predicted default')) # 27%

##### ROC, Lift Chart... etc.  #####
credit_model2_prob <- predict(credit_model2, credit_test, type="prob") #prob: set to 機率
head(credit_model2_prob)
library(ROCR)
pred <- prediction(predictions = credit_model2_prob[,"Yes"], labels = credit_test$Default)
perf <- performance(pred, measure = "tpr", x.measure = "fpr") #true positive rate, False positive rate
plot(perf, main = "ROC curve for credit_model2", lwd = 3, colorize=T)
abline(a=0, b=1, lwd=2, lty=2)

# And then a lift chart
perf <- performance(pred,"lift","rpp") # rpp: Rate of positive predictions
plot(perf, main="lift curve for credit_model2", colorize=T)
# And then a precision/recall graphs
perf <- performance(pred, measure="prec", x.measure="rec")
plot(perf, main="Precision/recall graphs for credit_model2", colorize=T)
# And then a sensitivity/specificity plots
perf <- performance(pred, measure="sens", x.measure="spec")
plot(perf, main="Sensitivity/specificity plots for credit_model2", colorize=T)

##### The End #####

