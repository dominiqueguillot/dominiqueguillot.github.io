library(datasets)

data(iris)
pairs(iris)

ind_train = sample.split(iris[,1], SplitRatio = 2/3)
train = iris[ind_train,]
test = iris[!ind_train,]

one_vs_all_logistic = function(trainY, trainX, testX)
{
  yvals = unique(trainY)
  n = dim(testX)[1]
  scores = matrix(0, nrow = n, ncol=length(yvals))
  mytest = data.frame(x=testX)
  for(i in 1:length(yvals))
  {
    train_Y_cpy = factor(ifelse(trainY == yvals[i], 1,0))
    mytrain = data.frame(x=trainX, y=train_Y_cpy)
    model = glm(y ~ ., family=binomial(link='logit'), data=mytrain, control = list(maxit = 100))
    scores[,i] = predict(model, mytest, type='response')
  }
  
  yhat = apply(scores, 1, which.max)
  return(list(yhat=yvals[yhat],scores=scores))
}

result = one_vs_all_logistic(train[,5], train[,-5], test[,-5])

accuracy = mean(result$yhat == test[,5])
print(accuracy)