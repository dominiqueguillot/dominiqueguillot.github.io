# Supervised vs Unsupervised learning

Two important type of problems commonly occur in data science: supervised learning and unsupervised learning. 

## Supervised learning

In supervised learning problems, an outcome variable guides the learning process.

We are provided with:

1. A set of input variables (predictors or independent variables).
2. A set of output variables (response or dependent variables).

The data is therefore *labelled*.

We want to use the input to predict the output. Our goal is to build a model to learn patterns in the dataset, and to use the model to predict unobserved values. 

**Important:** One should keep in mind that future predictions of the model are highly dependent on previously observed data. 
* If there is no or only a small amount of available data in a given region of the domain of interest, the model prediction can be very bad. 
* Since predictions are based on previous data, they reinforce previous trends and can lead to biased decisions.

```{figure} images/machine-learning3.jpg
---
width: 300px
---
Keep in mind that machine learning algorithms learn from previous data and tend to reinforce previous trends and biases.
```

Let us look at some examples of supervised learning problems. 

```{note}
Later in the course, you will have the opportunity to examine the datasets given below and to construct models to analyze them.
```

(E-handwritten-digits)=
### Example: Handwritten digits

```{figure} images/digits.png
---
width: 500 px
---
```

* You are provided a <a href="https://en.wikipedia.org/wiki/MNIST_database" target="_blank">dataset</a> containing images (16 x 16 grayscale images say) of digits.
* Each image contains a single digit. 
* Each image is labelled with the corresponding digit. 
* After reshaping, you can think of each image as a vector in $X \in \mathbb{R}^{256}$ and the label as a scalar $Y \in \{0,\dots,9\}$.

**Idea:** with a large enough dataset, we should be able to *learn* to identify/predict digits automatically.

```{note}
A possible application of such a model is to read zip codes on envelopes and automatically sort mail. More sophisticated versions with datasets containing all letters of the alphabet can also be used to construct models that can convert handwritten text to digital text.
``` 

(E-gene-expression)=
### Example: Gene expression data

DNA microarrays can be used to measure the <a href="https://en.wikipedia.org/wiki/Gene_expression" target="_blank">expression</a> of a gene in a cell -- see figure below (rows = genes, columns = sample).

```{figure} images/gene_expression.png
---
width: 300 px
---
ESL, Figure 1.3.
```

Suppose the gene expression of millions of genes are measured for cells from patients with and without a certain type of cancer. (Note how the data is *labelled*, i.e., we know if each sample is associated to cancer or not.) 

Can a model be constructed to predict if a patient has the cancer based on their gene expression? Note that such a model may be able to detect the cancer before signs of the cancer start to appear. 

### Example: Spam data

* A dataset contains information from 4601 email messages, in a study to screen email for "spam" (i.e., junk email).
* The data was donated by George Forman from Hewlett-Packard laboratories.

```{figure} images/ESL_Table1p1.png
--- 
width: 600 px
---
ESL, Table 1.1.
```

* Each message is labelled as spam/email.
* Each message contains characteristics such as the count of different words and symbols.

We would like to build a model that automatically identifies if a given email is spam or not. 

```{note}
The common feature in supervised learning is that the data is labelled. Obtaining labelled data of good quality can be tedious and expensive, especially if it involves the intervention of a human to perform the labelling. This is a serious limitation to supervised learning.
```

## Unsupervised learning

In unsupervised learning problems, we observe only features. There is therefore **no dependent variable** (or label). Our goal is to detect structure, patterns, etc..

### Example: clustering

A typical unsupervised learning problem is to automatically detect *clusters* in data. 

For example, consider the following two-dimensional data: 
```{figure} images/clusters_bw.png
--- 
width: 300 px
---
```

One very naturally groups (or *clusters*) the data. The figure below was produced with a computer algorithm that labelled the data points according to the group structure present in the data:
```{figure} images/clusters.png
--- 
width: 300 px
---
```

Notice that the original data points have no labels. Nevertheless, the cluster structure can reaveal a lot of information about the data. For example, in a database containing customers' information, customers with similar characteristics or behavior may create clusters. One can then try to identify what are the common characteristics of customers in a given cluster. This can yield significant insights about customer behavior. 

Another nice application of clustering is image segmentation, where the goal is to divide an image into meaningful parts (e.g., separate objects). 

```{figure} images/image_seg.jpg
---
width: 300 px
---
```

Once relevant "parts" of an image have been discovered, one can try to automatically label them with the computer as in the image above. 

## The curse of dimensionality

In modern problems, we typically have to work with a very large number of variables (also called *features*). For example, the customer database of a large insurance company may contain dozens of features of each customer (e.g., their age, income, number of recent accidents, the model of car they own, their car value, etc.). Recall that future predictions made by a model heavily dependent on previous data. As a result, in order to make accurate predictions, it is important to have observations (also called *samples*) that cover the possible range of the features very well. It is important to realize that, as the number of features increases, the number of samples required to get a good coverage **dramatically increases**. 

### Example: data in 1-D and 2-D

Consider a dataset where only one feature is measured. For example, let us say we are trying to determine if a person has a cancer based on the gene expression of a single one of their genes (see the {ref}`E-gene-expression` above). Assume the gene measurements lie in the interval $[0,1]$. Suppose the following data is available (where $0=$no cancer, and $1=$cancer): 

|Gene #1| Cancer|
|-------|-------|
|0.0 | 0 |
|0.1 | 0 | 
|0.2 | 0 |
|0.3 | 1 | 
|0.4 | 1 | 
|0.5 | 1 | 
|0.6 | 0 | 
|0.7 | 0 | 
|0.8 | 1 | 
|0.9 | 1 |
|1.0 | 0 |


Representing the data on the real line, one may be fairly confident to have enough coverage of the interval $[0,1]$ to be able to be able to capture which regions are more prone to representing cancer or not.
```{figure} images/no-cancer.png
---
width: 700 px
---
```
Here 11 samples were required so that every point in the interval $[0,1]$ is at a distance at most $0.05$ from one of the samples.  

Now, let us see how the situation changes when 2 genes are measured: 

```{figure} images/no-cancer-2d.png
---
width: 700 px
---
```

Observe how we now need 121 points on a $0.1 \times 0.1$ grid to mimic the coverage we had in 1-dimension. Also observe that the coverage is not even as good: points in the middle of every $0.1 \times 0.1$ "square" are at a distance $\sqrt{0.05^2 + 0.05^2} = 0.05 \sqrt{2} \approx 0.0707$ from any of the samples. 

More generally, in dimension $d$, a "hyper-grid" with sidelength $0.1$ in the hypercube $[0,1]^d$ contains $11^d$ grid points. For example, with $d = 50$ (still a small number of genes to try to predict cancer), the number of grid points is about $4.177 \times 10^{15}$. As a fun comparison, this is rougly the number of grains of sand in a $65$ cubic meter of sand! This is also about $522,500$ times the 2025 population of the Earth! There is no way we can collect so many samples to solve our cancer problem...

### The curse of dimensionality phenomemon
The above illustrates that, as the number of features increases, the volume of the space grows so fast that data becomes sparse (i.e., the data does not cover the space well at all). This phenomenon is known as the **curse of dimensionality**. Without any supplementary structure, one would expect needing a number of samples exponential in the number of features in order to build a model that makes good prediction. Thankfully, in the real world, there is typically structure that we can exploit to construct good models without needing an excessive amount of data. 

```{note}
In mathematics, a *sparse* vector is a vector that contains a lot of zero entries. The *sparsity* of a vector is the percentage of its entries that are equal to $0$. Similarly, a sparse matrix is a matrix with a large percentage of its entries equal to $0$. In contrast, a matrix that does not contain any significant number of zero entries is said to be *dense*. Many numerical algorithms in linear algebra can be developed to exploit sparsity and efficiently solve problems in very high dimensions. One example is sparse linear solvers. A typical laptop in 2025 can solve a system of equations with ten millions of unknowns and about 10 nonzeros per row in a few seconds to a few minutes. In contrast, an algorithm that doesn't exploit the sparsity of the system can take days or even weeks to solve the same system on the same machine.   
```

### Example: linear regression

In a linear regression model, one attempts to predict the dependent variable $Y$ using a *linear combination* of the predictors $X_1, X_2, \dots, X_p$: 

$
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_p X_p.
$

In traditional statistics, it is common to assume the number of samples available to observe this relationship (usually denoted by $n$) is much larger than the number of variables ($p$), and is at least greater than $p$. However, in many modern problems, the number of variables is often much larger than the number of samples. Think for example of the gene expression problem ({ref}`E-gene-expression`), where there are millions of genes a researcher may want to consider to detect a cancer, but there is only a small number of patients available in the study (e.g., 100). Here, one needs to make use of additional structure in order to be able to build a good model. In that case, it is natural to assume only a handful of genes are good predictors for a given cancer (i.e., many of the regression coefficients $\beta_i$ are equal to $0$). The problem though is we do not know which ones are non-zero. As we will see in future chapters, there are well-known methods (e.g., LASSO regression) that attempt to detect non-zero coefficients in linear regression. Such methods can be used to build linear models even when the sample size is much smaller than the number of variables.  

```{admonition} Assignment
---
class: warning
---
Please complete Homework 1 on <a href="https://sites.udel.edu/canvas/" target="_blank">Canvas</a> before continuing to read.
```