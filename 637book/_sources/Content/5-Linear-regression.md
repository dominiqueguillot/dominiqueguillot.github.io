# Linear Regression

The first family of models we will study are linear regression models. While such models may seem elementary at first sight, they are very useful and can often be used as a baseline before trying more sophisticated techniques. Another important advantage of linear models is that they are easy to **interpret**. In many situations, more complicated models (e.g., deep learning) may give better results, but provide no explanation of how they get their results. When working with large datasets, some of these models may also able to infer information such as gender and race, and use it indirectly. In some areas in the industry, this is a deal breaker. For example, the way mortgages are written is very regulated. Only specific information can typically be used to accept or decline a loan. On the other hand, in other areas (e.g., quantitative finance) building a model that produces excellent results can be the only thing that matters. Still, in general, being able to understand how a model arrived at a certain prediction is typically very important and desirable. 

```{figure} images/black-box.png
---
width: 300 px
---
Knowing how a model reaches a certain conclusion is often desirable. Very complex models tend to act like black boxes.
```

## The linear regression model

In linear regression, we attempt to predict a dependent variable $Y$ using a linear combination of predictors $X_1, X_2, \dots, X_p$:

$$
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_p X_p. 
$$(E-lin-reg)

The coefficients $\beta_0, \beta_1, \dots, \beta_p$ are called the *regression coefficients*. The first coefficient, $\beta_0$, is called the *intercept* of the model. 

In order to train a linear regression model, one needs several simultaneous observations of $Y$ and of $X_1, \dots, X_p$. We therefore assume several *training pairs* $(y_i, \mathbf{x_i})$ are provided to train the model, where $\mathbf{x_i} = (x_{i,1}, x_{i,2}, \dots, x_{i,p}) \in \mathbb{R}^p$ are the feature observations that correspond to the output $y_i$ for $i=1,\dots, n$. Let $f(x) = \beta_0 + \beta_1 x_1 + \dots + \beta_p x_p$. Given the training data, our goal is to estimate the values of the regression coefficients so that $y_i \approx f(x_i)$ for all $i=1,\dots,n$. 

### Example: predicting car prices

The file [JSE_Car_Lab.csv](data/JSE_Car_Lab.csv) contains the price of $n=805$ cars as well as $p=11$ features for each car. The data was compiled by Kelley Blue Book.  

```{figure} images/data-cars.jpg
---
width: 800 px
---
```

It is very natural to try to predict the price of the cars using a linear combination of the different numerical features. For example, one might expect the car mileage to play a significant role in predicting the value of the car. 

In order to use the non-numerical features (e.g., the car make), one needs to *encode* them in some way. A popular way to encode such variables is to use a **one-hot encoding**. For example, suppose there are only 3 possible car makes (say, Acura, Buick, and Chevrolet). It would be tempting to encode these car makes directly using a number (e.g., 1 = Acura, 2 = Buick, 3 = Chevrolet). However, in a linear model, doubling the value of a variable will double its impact on the price. This type of relationship cannot be justified here. Instead, a one-hot encoding adds three columns to the dataset, where each column indicates whether the car is of a given brand (1) or not (0). Thus, each row has precisely one "1" that indicates the make of the car. 

|Acura| Buick | Chevrolet|
|-----|-------|----------|
|1|0|0|
|0|1|0
|0|0|1|

For example, in the table above, the first row represents an Acura, the second row a Buick, and the third row a Chevrolet. Notices that this approach replaces the make column by $3$ new columns (the new value of $p$ is now $13$). 

Using a one-hot encoding of non-numerical variables has the advantage of treating each possible value of the variable equally. One downside thought is that the number of features of the data can increase significantly if the variable takes a lot of different values (since one column is added for each possible value). 

In the next chapter, you will analyze the JSE_Car_Lab dataset and will construct your own linear models to predict the price of cars. 

```{note}
Constructing a model for the price of the cars is very useful. Once a good model has been constructed, it can be used to estimate the market value of **any** car. This provides very useful information to car buyers and sellers.   
```


```{admonition} Removing the intercept
:name: N-lin-reg-no-intercept

When working with a linear model, we can always append a column whose entries are all $1$ to the dataset, and discard the intercept from the model. Observe that by doing that, the regression coefficient $\beta_1$ becomes the intercept. This simple modification makes it easier to make calculations with linear models and will be used below.
```

(sec-loss-function)=
### Loss functions

In order to measure how well a model is doing, we need to fix a way to measure the error made by the model on a given dataset. Such a measure of error is called a **loss function**. 

When predicting continuous data (as in the cars problem above), a loss function that is commonly used is the *mean squared error* (MSE): 

$$
MSE(\beta_0, \beta_1, \dots, \beta_p) = \frac{1}{n} \sum_{i=1}^n (y_i - f(\mathbf{x_i}))^2. 
$$

Note that the MSE depends on the regression coefficients since $f$ is a function of $\beta_0, \beta_1, \dots, \beta_p$. 

When predicting an output variable taking finitely many possible values the *cross-entropy* loss function is typically used. We will discuss cross-entropy and loss functions in the chapter on [](C-categorical).

When fitting a model, the goal is to minimize the loss function in order for the model to match the data as best as possible. To find the optimal regression coefficients in Equation {eq}`E-lin-reg`, we therefore need to solve the optimization problem: 

$$
\min_{\beta_0, \beta_1, \dots, \beta_p \in \mathbb{R}} L(\beta_0, \beta_1, \dots, \beta_p) = \min_{\beta_0, \beta_1, \dots, \beta_p \in \mathbb{R}} \frac{1}{n} \sum_{i=1}^n (y_i - f(\mathbf{x_i}))^2,
$$(E-lin-reg-optim)
where $f(\mathbf{x}) = \beta_0 + \beta_1 x_1 + \dots + \beta_p x_p$ and $\mathbf{x} = (x_1, x_2, \dots, x_p)$. 

(sec-finding-optimal-coefficients)=
### Finding the optimal coefficients

There are different approach to solve the optimization problem {eq}`E-lin-reg-optim`.

Recall that when a smooth functions of several variables reaches a local minimum at an interior point of its domain, its gradient has to be the zero vector. We will therefore compute the gradient of the above loss function and set it to zero to find candidates for the minimum. 

To simplify notation, let $y$ be the vector contains all the outputs of the regression, and let $X$ be the matrix with rows $\mathbf{x_1}^T, \mathbf{x_2}^T, \dots, \mathbf{x_n}^T$: 

$$
y = (y_1, y_2, \dots, y_n)^T \qquad X = \begin{pmatrix}
\mathbf{x_1} \\
\mathbf{x_2} \\
\vdots \\
\mathbf{x_n}
\end{pmatrix}.
$$

Notice that $X = (x_{i,j})$ is an $n \times p$ matrix and $y \in \mathbb{R}^n$. To simplify the calculations, we will assume the first column of $X$ has all its entries equal to $1$ and the linear model has no intercept (See the {ref}`(Removing the intercept)<N-lin-reg-no-intercept>` box).

Observe that the loss function can be re-written using a matrix vector product:

$$
L(\beta_1, \dots, \beta_p) = \frac{1}{n} \|y - X\beta\|_2^2, 
$$

where $\|\mathbf{z}\|_2^2 = \sum_{i=1}^n z_i^2$ for $z \in \mathbb{R}^n$. Expanding the expression, we obtain 

$$
L(\beta_1, \dots, \beta_p) = \frac{1}{n} \sum_{i=1}^n \left(y_i - \sum_{j=1}^p x_{i,j} \beta_j\right)^2.
$$

For $k=1,\dots,p$, differentiating with respect to $\beta_k$  yields

$$
\frac{\partial L}{\partial \beta_k} = -\frac{2}{n} \sum_{i=1}^n x_{i,k} \left(y_i - \sum_{j=1}^p x_{i,j} \beta_j\right). 
$$

Finally, setting the gradient to $0$, we obtain: 

$$
\sum_{i=1}^n x_{i,k} y_i = \sum_{i=1}^n \sum_{j=1}^p x_{i,k} x_{i,j} \beta_j \qquad (k=1,\dots,p).
$$(E-normal)

```{admonition} Matrix-vector and Matrix-matrix products

Recall that when we multiply a matrix $A = (a_{i,j}) \in \mathbb{R}^{n \times p}$ by a vector $x = (x_1, \dots, x_p)^T \in \mathbb{R}^p$, the $k$-th entry of $Ax$ is: 

$$
(Ax)_k = \sum_{j=1}^p a_{k,j} x_j. 
$$

Similarly, if $B$ is a $p \times r$ matrix, then the $(k,l)$-th entry of the matrix product $AB$ is

$$
(AB)_{k,l} = \sum_{j=1}^p a_{k,j} b_{j,l}. 
$$

Finally, recall that the $(k,l)$ entry of $A^T$ is $a_{l,k}$. 
```

Now, Equation {eq}`E-normal` can be written in a better way using matrix-vector and matrix-matrix products. First observe that the left-hand side is the $k$-th entry of $X^T y$. Similarly, the right-hand side is the $k$-th entry of $X^TX \beta$, where $\beta = (\beta_1, \dots, \beta_p)^T$. We therefore conclude that the gradient of the loss function is $0$ if and only if: 

$$
\boxed{X^TX \beta = X^T y}.
$$

The above linear system is called the **normal equations** associated to the linear regression problem. Since the loss function has to admit a minimum at some $\beta \in \mathbb{R}^p$, we are guaranteed that the normal equations have at least one solution. However, unless $X^TX$ has full rank (and is therefore invertible), the solution may not be unique. However, with some extra work, one can show that each solution of the normal equations achieves the minimum value of the loss function. We can therefore solve the normal equations to obtain all the minima of the loss function.

When $X^TX$ is invertible, the unique minimum of the loss function is:

$$
\boxed{\widehat{\beta} = (X^TX)^{-1} X^T y}.
$$

We often call $\widehat{\beta}$ the *least squares* estimator as it minimizes the sum of squares of the error. 

In conclusion, the above work shows that finding the optimal coefficients in linear regression is equivalent to solving a linear system of equations. In the next chapter, we will see how this can be done with Python.
