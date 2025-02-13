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
n <- nrow(iris2)

pch1 <- rep(".",n)
pch1[outliers_kmeans] <- "+"
col1 <- rep("black",n)
col1[outliers_kmeans] <- "red"
pairs(iris2, pch = pch1, col = col1)

