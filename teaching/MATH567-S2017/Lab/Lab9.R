library(tree)
library(caTools)

dat = read.table('./data/titanic/titanic_train.csv', header=TRUE, sep=',')
dat$Survived = as.factor(dat$Survived)
dat$Pclass = as.factor(dat$Pclass)

ind_train = sample.split(dat[,1], SplitRatio = 2/3)

train = na.omit(dat[ind_train,])
test = na.omit(dat[!ind_train,])

tree.titanic = tree( Survived~Sex+Age+Pclass+Fare , train)
summary(tree.titanic)

plot(tree.titanic)
text(tree.titanic, pretty=0)

yhat = predict(tree.titanic, test, type="class")

error = mean(test$Survived != yhat)
print(c("Classification tree error: ", error))

cv.titanic = cv.tree(tree.titanic, FUN = prune.misclass)
ind_min = which.min(cv.titanic$dev)
prune.titanic = prune.misclass (tree.titanic , best =cv.titanic$size[ind_min])

yhat_prune = predict(prune.titanic, test, type="class")
error_prune = mean(test$Survived != yhat_prune)
print(c("Pruned tree error: ", error_prune))

# Bagging
library(randomForest)

bag.titanic = randomForest(Survived ~ Sex+Age+Pclass+Fare+SibSp, data=train, mtry = 4, importance=TRUE)
yhat.bag = predict(bag.titanic, test)

error_bag = mean(test$Survived != yhat.bag)
print(c("Bagging error: ", error_bag))
