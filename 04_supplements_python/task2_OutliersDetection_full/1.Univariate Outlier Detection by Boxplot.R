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