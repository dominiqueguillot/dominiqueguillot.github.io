library(ISLR)
data(Auto)
# Normal equations
model <- lm(Auto$mpg ~ Auto$horsepower + Auto$weight)

X = as.matrix(Auto[,c(4,5)])
Xp = cbind(matrix(1,392,1), X)

y = Auto[,1]

solve(t(Xp) %*% Xp, t(Xp) %*% y) 
# Training and test set


Auto <- Auto[,-9]  # Remove the "names" column

n <- dim(Auto)[1]

ntrain <- floor(0.75*n)
ntest <- n - ntrain

train_ind <-  sample(1:n, ntrain)

train <-  Auto[train_ind,]
test <-  Auto[-train_ind,]

model_full <- lm(mpg ~ ., data=train)
mean((predict(model_full, test[-1]) - test[,1])**2)

model <- lm(mpg ~ ., data=train[,append(c(5,7,8),1)])
mean((predict(model, test[,c(5,7,8)]) - test[,1])**2)

# Fit with all variables
MSE = matrix(NA,6,35)

for (i in 2:7){
  print(i)
  S = combn(2:8, i, simplify = FALSE)
  for(j in 1:length(S)){
    model <- lm(mpg ~ ., data=train[,append(S[[j]],1)])
    MSE[i-1,j] <- mean((predict(model, test[,S[[j]]]) - test[,1])**2)
  }
}

plot(rep(2,35), MSE[1,], xlim=c(1,8), ylim=c(10,45), xlab="Size", ylab="MSE")
for(i in 3:7){
  par(new=T)
  plot(rep(i,35), MSE[i-1,], xlim=c(1,8), ylim=c(10,45), xlab="Size", ylab="MSE")
}

m <- apply(MSE, 1, min, na.rm = TRUE)

plot(2:7, m, xlab="Size of subset", ylab="Best subset test MSE", pch=16)

library(leaps)
fwd = regsubsets(mpg ~ ., data=Auto[,-9], method="forward")
bwd = regsubsets(mpg ~ ., data=Auto[,-9], method="backward")

