library(class)

tmp_train = read.table("zip.train", header = FALSE, sep = " ")

N = 3000
train.y = factor(tmp_train[1:N,1])
train.x = tmp_train[1:N,2:257]

tmp_test = read.table("zip.test", header = FALSE, sep = " ")

test.y = factor(tmp_test[,1])
test.x = tmp_test[,2:257]

knn_pred = knn(train.x, test.x, train.y, k=5)
error = test.y != knn_pred
error_rate_knn = sum(error)/length(error)*100

library(caret)

ctrl <- trainControl(method="repeatedcv", number=10, repeats = 1)
fitKnn = train(train.x, train.y, method="knn", trControl = ctrl, tuneGrid=expand.grid(.k=1:10), metric="Accuracy") 

knn_pred = knn(train.x, test.x, train.y, k=5)
error = test.y != knn_pred
error_rate_knn = sum(error)/length(error)*100