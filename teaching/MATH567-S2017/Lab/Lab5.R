# Multivariate normal

library(mvtnorm)

mu = rep(0,2)

X = matrix(runif(4), nrow=2)

S = t(X) %*% X

x = seq(-1,1,by=0.05)
y = seq(-1,1,by=0.05)
n = length(x)

z = matrix(0,nrow=n, ncol=n)

for(i in 1:n)
{
  for(j in 1:n)
  {
    z[i,j] = dmvnorm(c(x[i],y[j]),mean=mu, sigma=S)
  }
}

contour(x,y,z)
e = eigen(S)
arrows(0,0,e$vectors[1,1], e$vectors[2,1])
arrows(0,0,e$vectors[1,2], e$vectors[2,2])

#persp(x,y,z,theta = 30, phi = 30)

# Titanic
data = read.table('./titanic_train.csv', header=TRUE, sep=',')
#test = read.table('./titanic_test.csv', header=TRUE, sep=',')

library(caTools)
ind_train = sample.split( data[,1], SplitRatio = 2/3)

train = data[ind_train,]
test = data[!ind_train,]
model = glm(Survived ~ Pclass + Sex + Age + Embarked + SibSp,family=binomial(link='logit'),data=train)

yhat = predict(model, test, type='response')

isna = is.na(yhat)
ypred = as.numeric(yhat[!isna] > 0.5)
mean(ypred == test[!isna,"Survived"])
