# Distribution based outlier detection for univariate data ####
# install.packages("extremevalues")
library(extremevalues)
y <- rlnorm(100)
y <- c(0.1*min(y),y,10*max(y))
?getOutliers
K <- getOutliers(y, method = "I", distribution = "lognormal")
L <- getOutliers(y, method = "II", distribution = "lognormal")
op <- par(mfrow = c(1,2))
outlierPlot(y, K, mode = "qq")
outlierPlot(y, L, mode = "residual")
par(op)

# install.packages("mvoutlier")
library(mvoutlier)
set.seed(1234)
x <- cbind(rnorm(80), rnorm(80))
y <- cbind(rnorm(10, 5, 1), rnorm(10, 5, 1))
z <- rbind(x,y)
# Anomaly detection for one-dimensional data 一維資料的異常檢測
res1 <- uni.plot(z)
# Return the outliers' index 返回離群值編號
which(res1$outliers == T)
# Anomaly detection based on robust Mahalanobis distance 基於穩健馬氏距離的多元異常值（離群值）檢驗
res2 <- aq.plot(z)
# Return the outliers' index 返回離群值編號
which(res2$outliers == T)

# Anomaly detection for high-dimensional data 在高維空間中的異常值檢驗
data(swiss)
res3 <- pcout(swiss)
# Return the outliers' index 返回離群值編號
which(res3$wfinal01 == 0)


# install.packages("outliers")
library(outliers)

#outlier(x, opposite = FALSE, logical = FALSE)
set.seed(1234)
y = rnorm(100)
outlier(y)
outlier(y, opposite = TRUE)
dim(y) <- c(20,5)
outlier(y)
outlier(y, opposite = TRUE)

#grubbs.test(x, type = 10, opposite = FALSE, two.sided = FALSE)
set.seed(1234)
x = rnorm(10)
grubbs.test(x)
grubbs.test(x, type = 10)
grubbs.test(x, type = 11)
grubbs.test(x, type = 20)

# set.seed(1234)
# x=rnorm(100)
# d=data.frame(x=x,group=rep(1:10,10))
# cochran.test(x~group,d)
# cochran.test(x~group,d,inlying=TRUE)
# x=runif(5)
# cochran.test(x,rep(5,5))
# cochran.test(x,rep(100,5))