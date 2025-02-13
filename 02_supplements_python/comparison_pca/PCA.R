cell <- read.csv('cell_num.csv')[-1]

cell.pca <- prcomp(cell, center = TRUE, scale. =TRUE)
cell.pca$rotation
cell.pca$x

summary(cell.pca)

library(devtools)
# install_github("vqv/ggbiplot")
library(ggbiplot)

ggbiplot(cell.pca)
ggbiplot(cell.pca, labels=rownames(cell)) 

# loading and score matrix 


cell.pca2 <- princomp(cell)
cell.pca2$loadings
cell.pca2$scores
