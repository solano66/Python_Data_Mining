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
