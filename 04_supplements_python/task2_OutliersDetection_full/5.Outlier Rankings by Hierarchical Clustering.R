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
