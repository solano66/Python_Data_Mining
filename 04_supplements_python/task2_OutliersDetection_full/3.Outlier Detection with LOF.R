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
tmp <- density(outlier.scores) # generic function density computes kernel density estimates
names(tmp) # x, y, bw, n, call, data.name, has.na

# in R $ operator can be used to select a variable/column, assign new values to a variable/column, 
# or add a new variable/column in an R object

#names(iris2)
# tmp$x # over at least 512 points
# tmp$y # over at least 512 points
# tmp$bw
# tmp$n
# tmp$call
# tmp$data.name
# tmp$has.na  # is to check is there any na value in python is None, null or nill in other program language

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
