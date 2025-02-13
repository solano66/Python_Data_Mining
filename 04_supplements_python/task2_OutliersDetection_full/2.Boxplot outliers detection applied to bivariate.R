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
