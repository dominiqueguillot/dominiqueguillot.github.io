library(glmnet)

# Generate data
n <- 100   # Sample size
p <- 500   # Nb. of variables

true_p <- 10

X <- matrix(rnorm(n*p), nrow=n, ncol=p)

true_beta = matrix(rep(0,p), nrow=p)
true_beta[1:10] = 1

SNR <- 1  # Signal-to-noise ratio = ratio of variances

noise <- matrix(rnorm(n, sd=1/sqrt(SNR)),nrow=n)

y <- X%*%true_beta + noise

fit.ridge <- glmnet(X, y, family="gaussian", alpha=0)

matplot(fit.ridge$lambda, t(fit.ridge$beta[1:10,]),type='l')
matplot(fit.ridge$lambda, t(fit.ridge$beta[11:20,]), type='l')

fit.lasso <- glmnet(X, y, family="gaussian", alpha=1)

matplot(fit.lasso$lambda, t(fit.lasso$beta[1:10,]),type='l')
matplot(fit.lasso$lambda, t(fit.lasso$beta[11:20,]), type='l')


# SNR = 0.5
SNR <- 0.1  # Signal-to-noise ratio = ratio of variances

noise <- matrix(rnorm(n, sd=1/sqrt(SNR)),nrow=n)

y <- X%*%true_beta + noise

fit.lasso <- glmnet(X, y, family="gaussian", alpha=1)

matplot(fit.lasso$lambda, t(fit.lasso$beta[1:10,]),type='l')
matplot(fit.lasso$lambda, t(fit.lasso$beta[11:100,]), type='l')

# CV
SNR <- 1.0  # Signal-to-noise ratio = ratio of variances
noise <- matrix(rnorm(n, sd=1/sqrt(SNR)),nrow=n)
y <- X%*%true_beta + noise

cvlasso <- cv.glmnet(X, y, type.measure="mse", family="gaussian", alpha=1.0)
best_lasso <- glmnet(X,y, family="gaussian", alpha=1.0, lambda=cvlasso$lambda.min)
ind_select <- which(best_lasso$beta != 0)

lasso1se <- glmnet(X,y, family="gaussian", alpha=1.0, lambda=cvlasso$lambda.1se)

fit = lm(y ~ X[,ind_select])

# Exercise 3
load("Westbc.rda")

pheno <- matrix(rep(0.0,49), nrow=49, ncol=1)
pheno[Westbc$pheno == 'positive'] = 1.0

assay <- t(Westbc$assay)

ntrain = floor(2/3*49)
ntest = 49 - ntrain

train_ind = sample(1:49, ntrain)

x.train = assay[train_ind,]
x.test = assay[-train_ind,]

y.train = pheno[train_ind,]
y.test = pheno[-train_ind,]

cvlasso <- cv.glmnet(x.train, y.train, type.measure="mse", family="gaussian", alpha=1.0)
best_lasso <- glmnet(x.train,y.train, family="gaussian", alpha=1.0, lambda=cvlasso$lambda.min)
ind_select <- which(best_lasso$beta != 0)

data_train = as.data.frame(cbind(y.train, x.train[,ind_select]))
model = lm(y.train~., data = data_train)

data_test = as.data.frame(cbind(y.test, x.test[,ind_select]))
y_predict = predict(model, data_test)
y_pred = rep(0, ntest)
y_pred[y_predict > 0.5] = 1

error = sum(abs(y_pred - y.test))/ntest*100

# Repeat 100 times

error = rep(NA,100)
for(i in 1:100)
{
  print(i)
  train_ind = sample(1:49, ntrain)
  
  x.train = assay[train_ind,]
  x.test = assay[-train_ind,]
  
  y.train = pheno[train_ind,]
  y.test = pheno[-train_ind,]
  
  cvlasso <- cv.glmnet(x.train, y.train, type.measure="mse", family="gaussian", alpha=1.0)
  best_lasso <- glmnet(x.train,y.train, family="gaussian", alpha=1.0, lambda=cvlasso$lambda.min)
  ind_select <- which(best_lasso$beta != 0)
  data_train = as.data.frame(cbind(y.train, x.train[,ind_select]))
  model = lm(y.train~., data = data_train)
  
  data_test = as.data.frame(cbind(y.test, x.test[,ind_select]))
  y_predict = predict(model, data_test)
  y_pred = rep(0, ntest)
  y_pred[y_predict > 0.5] = 1
  
  error[i] = sum(abs(y_pred - y.test))/ntest*100
}
