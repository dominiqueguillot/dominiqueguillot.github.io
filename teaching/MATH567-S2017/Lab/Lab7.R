# Load data
tmp = read.table("zip.train", header = FALSE, sep = " ")
x_zip = tmp[,2:257]
y_zip = tmp[,1]

# Compute principal components
pr.out = prcomp ( x_zip , scale = TRUE )

# Plot first 2 principal components
library(grDevices)
mypal = rainbow(10)
plot(pr.out$x[,1:2], pch=20, col=mypal[y_zip+1])
legend("bottomright",legend=0:9, col=mypal[1:10], pch=20, ncol=2)
