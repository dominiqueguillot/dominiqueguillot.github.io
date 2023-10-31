# Exercise 1
##############################
# This is how the graph was generated...
p = 15
A = matrix(rbinom(p*p, 1, 0.15),p,p)
A[lower.tri(A, diag=FALSE)] = 0
A = A + t(A)
A[1:5,1:5] = 1
A[6:10, 6:10] = 1
A[11:15, 11:15] = 1
A = A-diag(p)

image(A)

# Shuffles the vertices
rperm = sample(1:p, p, replace=FALSE)
Ap = A[rperm, rperm]
save(Ap, file="graph.RData")
##############################

## Soln to Exercise 1##

load("graph.RData")
image(Ap)
library(igraph)
G = graph_from_adjacency_matrix(Ap, "undirected")
plot(G)
tkplot(G)

D = diag(rowSums(Ap))
L = D-Ap

e = eigen(L)
I = order(e$values)

eval = e$values[I]
evec = e$vectors[,I]

plot(evec[,2], evec[,3])
clus = kmeans(evec[,2:3], 3)

perm = order(clus$cluster)

Arec = Ap[perm, perm]
image(Arec)

# Exercise 2 
library(jpeg)
library(kernlab)

cat = readJPEG('scat.jpg')
gcat = readJPEG('scat_gray.jpg')
image(gcat)

p1 = dim(cat)[1]
p2 = dim(cat)[2]
p = p1*p2
catflat = matrix(cat, nrow=p)

sc = specc(catflat, centers=4)
Ccat = matrix(sc, nrow=p1, ncol=p2)
image(Ccat)