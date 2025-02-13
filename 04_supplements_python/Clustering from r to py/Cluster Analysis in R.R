#### Cluster Analysis in R ####
# source :
# https://girke.bioinformatics.ucr.edu/GEN242/tutorials/rclustering/rclustering/

library(gplots)
# the data
y <- matrix(rnorm(500), 100, 5, dimnames = list(paste0("g", 1:100, sep=""), paste("t", 1:5, sep="")))
heatmap.2(y) # Shortcut to final result

#### Stepwise Approach with Tree Cutting
## Row- and column-wise clustering 
hr <- hclust(as.dist(1-cor(t(y), method="pearson")), method="complete")
hc <- hclust(as.dist(1-cor(y, method="spearman")), method="complete") 
## Tree cutting
mycl <- cutree(hr, h=max(hr$height)/1.5); mycolhc <- rainbow(length(unique(mycl)), start=0.1, end=0.9); mycolhc <- mycolhc[as.vector(mycl)] 
## Plot heatmap 
mycol <- colorpanel(40, "darkblue", "yellow", "white") # or try redgreen(75)
heatmap.2(y, Rowv=as.dendrogram(hr), Colv=as.dendrogram(hc), col=mycol, scale="row", density.info="none", trace="none", RowSideColors=mycolhc) 


#### K-Means Clustering
km <- kmeans(t(scale(t(y))), 3)
km$cluster #<<++++ last here

#### Fuzzy C-Means Clustering
library(cluster) # Loads the cluster library.
fannyy <- fanny(y, k=4, metric = "euclidean", memb.exp = 1.2)
round(fannyy$membership, 2)[1:4,]

fannyy$clustering # Hard clustering result

(fannyyMA <- fannyy$membership > 0.20)[1:4,] # Soft clustering result

apply(fannyyMA, 1, which)[1:4] # First 4 clusters

#### Principal Component Analysis (PCA)
pca <- prcomp(y, scale=T)
summary(pca) # Prints variance summary for all principal components

plot(pca$x, pch=20, col="blue", type="n") # To plot dots, drop type="n"
text(pca$x, rownames(pca$x), cex=0.8)

#### Multidimensional Scaling (MDS)
loc <- cmdscale(eurodist) 
#write.csv(loc,"D:/others/loc.csv", row.names = FALSE)
plot(loc[,1], -loc[,2], type="n", xlab="", ylab="", main="cmdscale(eurodist)")
text(loc[,1], -loc[,2], rownames(loc), cex=0.8) 

#### Biclustering
source("http://faculty.ucr.edu/~tgirke/Documents/R_BioCond/My_R_Scripts/clusterIndex.R") 
library(cluster); y <- matrix(rnorm(5000), 1000, 5, dimnames=list(paste("g", 1:1000, sep=""), paste("t", 1:5, sep=""))); clarax <- clara(y, 49); clV1 <- clarax$clustering; clarax <- clara(y, 50); clV2 <- clarax$clustering 
ci <- cindex(clV1=clV1, clV2=clV2, self=FALSE, minSZ=1, method="jaccard")
ci[2:3] # Returns Jaccard index and variables used to compute it 

## Clustering cluster sets with Jaccard index
clVlist <- lapply(3:12, function(x) clara(y[1:30, ], k=x)$clustering); names(clVlist) <- paste("k", "=", 3:12)
d <- sapply(names(clVlist), function(x) sapply(names(clVlist), function(y) cindex(clV1=clVlist[[y]], clV2=clVlist[[x]], method="jaccard")[[3]]))
hv <- hclust(as.dist(1-d))
plot(as.dendrogram(hv), edgePar=list(col=3, lwd=4), horiz=T, main="Similarities of 10 Clara Clustering Results for k: 3-12") 


##### Clustering Excercises
## Data Preprocessing
# Scaling
## Sample data set
set.seed(1410)
y <- matrix(rnorm(50), 10, 5, dimnames=list(paste("g", 1:10, sep=""), 
                                            paste("t", 1:5, sep="")))
dim(y)

## Scaling
yscaled <- t(scale(t(y))) # Centers and scales y row-wise
apply(yscaled, 1, sd)

## Distance Metrices
# Euclidean distance matrix
dist(y[1:4,], method = "euclidean")

# Correlation-based distance matrix
c <- cor(t(y), method="pearson") 
as.matrix(c)[1:4,1:4]

# correlation-based distance matrix
d <- as.dist(1-c)
as.matrix(d)[1:4,1:4]

## Hierarchical Clustering with hclust
hr <- hclust(d, method = "complete", members=NULL)
names(hr)

par(mfrow = c(1, 2)); plot(hr, hang = 0.1); plot(hr, hang = -1) 

# Tree plotting I
plot(as.dendrogram(hr), edgePar=list(col=3, lwd=4), horiz=T) 

# Tree plotting II
library(ape) 
plot.phylo(as.phylo(hr), type="p", edge.col=4, edge.width=2, 
           show.node.label=TRUE, no.margin=TRUE)

# Tree Cutting
hr
## Print row labels in the order they appear in the tree
hr$labels[hr$order] 
# tree cutting with cutree
mycl <- cutree(hr, h=max(hr$height)/2)
mycl[hr$labels[hr$order]] 

#### Heatmaps
library(gplots)
heatmap.2(y, col=redgreen(75))
# with pheapmap
library(pheatmap); library("RColorBrewer")
pheatmap(y, color=brewer.pal(9,"Blues"))

## Customize heatmaps
hc <- hclust(as.dist(1-cor(y, method="spearman")), method="complete")
mycol <- colorpanel(40, "darkblue", "yellow", "white")
heatmap.2(y, Rowv=as.dendrogram(hr), Colv=as.dendrogram(hc), col=mycol,
          scale="row", density.info="none", trace="none", 
          RowSideColors=as.character(mycl))

## K-Means Clustering wit PAM
library(cluster)
pamy <- pam(d, 4)
(kmcol <- pamy$clustering)

heatmap.2(y, Rowv=as.dendrogram(hr), Colv=as.dendrogram(hc), col=mycol,
          scale="row", density.info="none", trace="none", 
          RowSideColors=as.character(kmcol))

## K-Means Fuzzy Clustering
library(cluster)
fannyy <- fanny(d, k=4, memb.exp = 1.5)
round(fannyy$membership, 2)[1:4,]

fannyy$clustering 

## Returns multiple cluster memberships for coefficient above a certain 
## value (here >0.1)
fannyyMA <- round(fannyy$membership, 2) > 0.10 
apply(fannyyMA, 1, function(x) paste(which(x), collapse="_"))

## Multidimensional Scaling (MDS)
loc <- cmdscale(eurodist) 
## Plots the MDS results in 2D plot. The minus is required in this example to 
## flip the plotting orientation.
plot(loc[,1], -loc[,2], type="n", xlab="", ylab="", main="cmdscale(eurodist)")
text(loc[,1], -loc[,2], rownames(loc), cex=0.8) 

## Principal Component Analysis(PCA)
library(scatterplot3d)
pca <- prcomp(y, scale=TRUE)
names(pca)

summary(pca) # Prints variance summary for all principal components.

scatterplot3d(pca$x[,1:3], pch=20, color="blue") 

sessionInfo()


#write.csv(mtcars,"D:/others/mtcars.csv", row.names = FALSE)
