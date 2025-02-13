### Outlier Detection ####
### Univariate Outlier Detection by Boxplot ####
set.seed(3147)
x <- rnorm(100);x     # it's generates a random number using a normal(bell curve) distribution
summary(x)            # it will summarize the statistical result like, mean, min, max, median
boxplot.stats(x)$out  #
str(boxplot.stats(x)) #
boxplot.stats(x)      #

# check what inside the boxplot.stats(x)
dat <- boxplot.stats(x) 
dat
rm(dat)

boxplot(x,notch = TRUE) # this is how the boxplot show up

### Boxplot outliers detection applied to bivariate case ####
y <- rnorm(100)
df <- data.frame(x,y)
rm(x,y) # for garbage collection # i think
head(df,4)

attach(df)
(a <- which(x %in% boxplot.stats(x)$out)) # it's sorting the data in boxplot.stats
(b <- which(y %in% boxplot.stats(y)$out))
detach(df)


(outlier.list1 <- intersect(a,b)) # prob intersection same number in a and b
plot(df)
points(df[outlier.list1,], col = 'red', pch = "+", cex = 2)

(outlier.list2 <- union(a,b))  # all in a and b
plot(df)
points(df[outlier.list2,], col = 'blue', pch = "+", cex = 1.5)

### Outlier Detection with LOF (local outliers factor) ####
# library('DMwR')   # the library was deleted from the official web
# https://cran.r-project.org/web/packages/DMwR/index.html
library('DMwR2')
iris2 <- iris[,1:4]
outlier.scores <- lofactor(iris2, k = 5) # k: number of neighbors
? lofactor
class(outlier.scores) # just a numeric vector of 150
outlier.scores
plot(density(outlier.scores)) # plot.density()
tmp <- density(outlier.scores)
names(tmp) # x, y, bw, n, call, data.name, has.na
# tmp$x # over at least 512 points
# tmp$y # over at least 512 points
# tmp$bw
# tmp$n
# tmp$call
# tmp$data.name
# tmp$has.na

# Decreasingly ordered and get the top five 
outliers_lof <- order(outlier.scores, decreasing = T)[1:5] # get the top five indices
iris[outliers_lof,]
plot(density(outlier.scores[-outliers_lof]))

# Decreasingly ordered 
outlier.scores[order(outlier.scores, decreasing = TRUE)]

# Increasingly ordered 
outlier.scores[order(outlier.scores, decreasing = FALSE)]

outliers_lof
outlier.scores[outliers_lof] # lof scores are more than 1.548281 (inclusive)
iris2[outliers_lof,]

n <- nrow(iris2)
labels <- 1:n
labels[-outliers_lof] <- '.' # use '.' as labels except outliers
biplot(prcomp(iris2), cex = .8, xlabs = labels)
?biplot.princomp # this is a method for the generic function biplot
# supplements on PCA
prcomp(iris2)
attributes(prcomp(iris2))
prcomp(iris2)$sdev
prcomp(iris2)$rotation
prcomp(iris2)$center
prcomp(iris2)$scale
prcomp(iris2)$x
summary(prcomp(iris2))

# scatterplot matrix with outliers
pch1 <- rep(".",n)
pch1[outliers_lof] <- "+"
col1 <- rep("black",n)
col1[outliers_lof] <- "red"
pairs(iris2, pch = pch1, col = col1)

### Outlier Detection by K-Means Clustering ####
iris2 <- iris[,1:4]
# iris2 <- iris[1:4] # same as above
kmeans.result <- kmeans(iris2, centers = 3)
str(kmeans.result)
kmeans.result$centers
kmeans.result$centers[2,] # check specific cluster center
kmeans.result$cluster # cluster number for each obs.
centers <- kmeans.result$centers[kmeans.result$cluster,]
centers # 150*4 numeric matrix, interesting !
str(centers)
distances <- sqrt(rowSums((iris2 - centers)^2))
distances # a numeric vector of 150 elements
outliers_kmeans <- order(distances, decreasing = T)[1:5]
?order
outliers_kmeans
iris2[outliers_kmeans,]

plot(iris2[,c("Sepal.Length","Sepal.Width")], pch = "o", col = kmeans.result$cluster, cex = 0.8)
# plot cluster centers
points(kmeans.result$centers[,c("Sepal.Length","Sepal.Width")], col = 1:3, pch = 8, cex = 1.5)

# plot outliers
points(iris2[outliers_kmeans, c("Sepal.Length", "Sepal.Width")], pch = "+", col = 4, cex = 2)

# scatterplot matrix with outliers
pch1 <- rep(".",n)
pch1[outliers_kmeans] <- "+"
col1 <- rep("black",n)
col1[outliers_kmeans] <- "red"
pairs(iris2, pch = pch1, col = col1)

### Outlier Rankings by Hierarchical Clustering ####
# This function uses hierarchical clustering to obtain a ranking of outlierness for a set of cases. The ranking is obtained on the basis of the path each case follows within the merging steps of a agglomerative hierarchical clustering method.

library(DMwR2)
## Some examples with algae frequencies in water samples
data(algae)

## Trying to obtain a reanking of the 200 samples
o <- outliers.ranking(algae)
# Warning message:
# In dist(data, method = clus$dist) : NAs introduced by coercion

o$rank.outliers
o$prob.outliers
sort(o$prob.outliers) # same as rank.outliers

## As you may have observed the function complained about some problem
## with the dist() function. The problem is that the algae data frame
## contains columns (the first 3) that are factors and the dist() function
## assumes all numeric data.
## We can solve the problem by calculating the distance matrix "outside"
## using the daisy() function that handles mixed-mode data, as show in
## the code below that requires the R package "cluster" to be available
library(cluster)
dm <- daisy(algae)
o <- outliers.ranking(dm)

## Now let us check the outlier ranking factors ordered by decreasing
## score of outlyingness
o$prob.outliers[o$rank.outliers]

## Another example with detection of fraudulent transactions
data(sales)

## trying to obtain the outlier ranks for the set of transactions of a
## salesperson regarding one particular product, taking into
## consideration the overall existing transactions of that product
head(sales)
s <- sales[sales$Prod == 'p1',c(1,3:4)]  # transactions of product p1
head(s)
tr <- na.omit(s[s$ID != 'v431',-1])      # all except salesperson v431
head(tr)
ts <- na.omit(s[s$ID == 'v431',-1])
head(ts)

o <- outliers.ranking(data=tr,test.data=ts,
                      clus=list(dist='euclidean',alg='hclust',meth='average'))
# The outlyingness factor of the transactions of this salesperson
o$prob.outliers


### Additional Examples of Outlier Detection: Bivariate Boxplot and Convex Hull Trimming #### 
# install.packages("MVA",type='source')
# Bivariate Boxplot
library('MVA')
library(help = MVA)
??USairpollution # {HSAUR2}
library(help = HSAUR2)
# demo("Ch-MVA")
# demo("Ch-Viz")
library('HSAUR2');?USairpollution
str(USairpollution)

mlab <- 'Manufacturing enterprises with 20 or more workers'
plab <- 'Population size (1970 census) in thousands'

plot(popul ~ manu, data = USairpollution, xlab = mlab, ylab = plab)
outcity <- match(lab <- c("Chicago","Detroit","Cleveland","Philadelphia"), rownames(USairpollution)) # return matching indices in rownames(USairpollution)
x <- USairpollution[,c("manu", "popul")]
bvbox(x, mtitle = "", xlab = mlab, ylab = plab)
text(x = x$manu[outcity], y = x$popul[outcity], label = lab, cex = 0.7, pos = c(2,2,4,2))

with(USairpollution, cor(manu, popul))
with(USairpollution, cor(manu[-outcity], popul[-outcity])) # Decreasing! Why?

# Convex hull trimming
?chull # {grDevices}
(hull <- with(USairpollution, chull(manu, popul)))
with(USairpollution, plot(manu, popul, pch = 1, xlab = mlab, ylab = plab))
with(USairpollution, polygon(manu[hull], popul[hull], density = 15, angle = 30))

with(USairpollution, cor(manu[-hull], popul[-hull])) # Decreasing a little bit !

# detectTC, detectLS : 時間數列 TC 與 LS 型態離群值偵測 ####
#
# 作者：淡江大學統計系 陳景祥 2010/2/20
# 
# 基本語法:    detectTC(arima物件)
#              detectLS(arima物件)
#
# 其中 arima 物件是經過 arima 函數處理過的輸出變數
#      其餘參數用法，請參考 TSA 套件的 detectAO 函數說明
#
# 註：detectTC 與 detectLS 修改自 TSA 套件的 detectAO 函數

detectTC = function (object, alpha = 0.05, delta = 0.7, cutoff = 0, robust = TRUE) 
{
  resid = residuals(object)
  
  piwt = ARMAtoMA(ar = -object$mod$theta, ma = -object$mod$phi, 
                  lag.max = length(resid) - 1)
  
  n.piwt = length(piwt)
  
  x = numeric(n.piwt)
  
  for (k in 1:n.piwt)
  {
    if (k == 1)
      x[k] = delta - piwt[1]
    else
    {
      sum = 0
      for (j in 1:(k-1)) sum = sum + delta^(k-j)*piwt[j]
      x[k] = delta^k - sum - piwt[k]           
    }
  }
  
  x = c(1,-1*x)
  
  omega = filter(c(0 * resid[-1], rev(resid)), filter = x, side = 1, method = "convolution")
  
  
  omega = omega[!is.na(omega)]
  
  rho2 = 1/cumsum(x^2)      
  omega = omega * rho2
  
  if (robust) 
    sigma = sqrt(pi/2) * mean(abs(resid), na.rm = TRUE)        
  else sigma = object$sigma2^0.5
  
  lambda2T = omega/sigma/sqrt(rho2)
  lambda2T = rev(lambda2T)
  
  if (cutoff < 0.5)
    cutoff = qnorm(1 - alpha/2/length(lambda2T))      
  
  out = abs(lambda2T) > cutoff
  ind = seq(lambda2T)[out]
  lambda2 = lambda2T[out]
  
  if (length(ind) != 0) 
    print(rbind(ind, lambda2))
  else print("No TC-outlier detected")
  
  invisible(list(lambda2 = lambda2, ind = ind))
}

detectLS = function (object, alpha = 0.05, cutoff = 0, robust = TRUE) 
{
  resid = residuals(object)
  
  piwt = ARMAtoMA(ar = -object$mod$theta, ma = -object$mod$phi, 
                  lag.max = length(resid) - 1)
  
  n.pi = length(piwt)
  
  x = numeric(n.pi)
  
  for (k in 1:n.pi)
  {
    sumw = sum(piwt[1:k])
    x[k] = 1 - sumw           
  }
  
  x = c(1,-1*x)
  
  omega = filter(c(0 * resid[-1], rev(resid)), filter = x, side = 1, method = "convolution")
  
  omega = omega[!is.na(omega)]
  
  rho2 = 1/cumsum(x^2)    
  
  omega = omega * rho2
  
  if (robust) 
    sigma = sqrt(pi/2) * mean(abs(resid), na.rm = TRUE)        
  else sigma = object$sigma2^0.5
  
  lambda2T = omega/sigma/sqrt(rho2)
  lambda2T = rev(lambda2T)
  
  if (cutoff < 0.5 )    
    cutoff = qnorm(1 - alpha/2/length(lambda2T))
  
  out = abs(lambda2T) > cutoff
  ind = seq(lambda2T)[out]
  lambda2 = lambda2T[out]
  
  if (length(ind) != 0) 
    print(rbind(ind, lambda2))
  else print("No LS-outlier detected")
  
  invisible(list(lambda2 = lambda2, ind = ind))
}

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

