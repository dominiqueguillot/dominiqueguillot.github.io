#### Exercise 1 ####

A = matrix(runif(16),4,4)
B = matrix(runif(16),4,4)

# Matrix product
A %*% B
# Entrywise product
A*B

# Determinant
det(A)

# Eigenvalues/eigenvectors
y <- eigen(A)

# Random vector (Normal entries)
b = rnorm(4)

solve(A,b)
Ainv <- solve(A)
Ainv %*% b

##### Exercise 2 ####
library(ISLR)
data(Auto)   #  # Can also use attach(Auto)
Auto[1,]  
Auto[,'mpg']
summary(Auto)
# plot
plot(Auto$weight, Auto$mpg)

# Histogram
hist(Auto$mpg)

pairs(Auto)
pairs(~mpg + horsepower + weight, Auto)

# Fit a linear model
model <- lm(Auto$mpg ~ Auto$horsepower + Auto$weight)
summary(model)

m <- lm(Auto$horsepower ~ Auto$weight)
plot(Auto$weight, Auto$horsepower)
abline(m)
