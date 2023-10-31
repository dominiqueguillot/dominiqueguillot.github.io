# Exercise 1

library(MASS)

d1 = mvrnorm(50, mu=c(-2,0), Sigma = diag(2)*0.2)
d2 = mvrnorm(50, mu=c(2,0), Sigma = diag(2)*0.2)
data = rbind(d1,d2)

plot(data, asp=1)
c = kmeans(data,2)

print(c$centers)
print(c$withinss)
print(c$betweenss)

# Exercise 2

library(kernlab)
c = kmeans(spirals,2)

library(grDevices)
mypal = rainbow(2)
plot(spirals, col=mypal[c$cluster])

sc = specc(spirals, centers=2)
plot(spirals, col=mypal[sc],asp=1)
