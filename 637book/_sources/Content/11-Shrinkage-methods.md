# Improving linear regression

We saw before that the least squares estimator is the best linear unbiased estimator for the regression coefficients. We will now look at some *biased* estimators that often result in a better [bias-variance trade-off](S-bias-variance) and, therefore, in a lower MSE.

(sec-subset-selection)=
## Subset selection

Consider the linear regression model 

$$
Y = \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_p X_p.
$$

Our goal is to estimate the coefficients $\beta_1, \dots, \beta_p$. When the number of variables is large, one may expect several of the variables to not contribute significantly to predicting $Y$. For example, in the [cars dataset](sec-lab1-cars), some of the variables may not be good predictors for the price of the cars. Similarly, in the [gene expression example](E-gene-expression), most of the genes considered are probably unrelated to a given cancer. The difficulty in such problems is to identify which predictors are relevant. 

```{note}

In terms of getting the smallest *training error* possible, the model with the most variables will always do better. However, in terms of *testing error* a model with a smaller number of variables will often perform better. 
```

A first (perhaps naive) approach is to try all possible subsets of predictors and measure their test error. Recall that a set of size $p$ has $2^p$ subsets (each element can be picked or not picked, leading to 2 choices per element).  Quite surprisingly, the leaps and bounds procedure (<a href="https://www.jstor.org/stable/1267601" target="_blank">Furnival and Wilson</a>, 1974) makes it feasible to perform the regression for all subsets for $p$ as large as $30$ or $40$ by using a clever implementation that avoids computing things several times. However, when $p$ is larger, evaluating the test error associated to all subsets of predictors quickly becomes impossible. 

```{admonition} Exercise

Split the cars dataset from Lab 1 into a training and a testing set. You may want to restrict the predictors to the numerical ones for simplicity. For each subset of predictors, fit a linear model on the training set, and save its error on the test set. Use this to reproduce the figure below and find the subset of predictors that yields the best test error.

The following code may be useful to loop over subsets:

```python
import itertools
def findsubsets(S,m):
    return set(itertools.combinations(S, m))
    
for k in range(7):
    S = list(findsubsets(range(7),k+1))
    sets.append(S)
    
    for l in range(len(S)):
        # Fit model on training data for given subset
        scores[k,l] = ...    # Compute test score
```

The following figure displays the test error on the cars dataset for all possible subsets of the numerical predictors.

```{figure} images/cars-best-subset.png
---
width: 500 px
---
Test error for subset selection on the cars dataset. 
```

## Shrinkage methods

In order to restrict the number of nonzero regression coefficients or their size, a popular approach is to add a penalty (or "price to pay") for including a nonzero coefficient. For example, let $\lambda > 0$ be a parameter and consider the following problem:

$$
\hat{\beta}^0 = \textrm{argmin}_{\beta \in \mathbb{R}^p} \left(\|y - X\beta\|_2^2 + \lambda \sum_{i=1}^p {\bf 1}_{\beta_i \ne 0}\right).
$$

Here, 

$$
{\bf 1}_{\beta_i \ne 0} = \begin{cases}
1 & \textrm{ if } \beta_i \ne 0 \\
0 & \textrm{ otherwise}.
\end{cases}
$$

Observe how each nonzero coefficient $\beta_i$ adds $\lambda$ to the objective function to minimize. When optimizing over $\beta \in \mathbb{R}^p$, there is thus a "competition" between minimizing the error $\|y-X\beta\|^2$ and the price $\lambda$ paid for including each variable. When $\lambda$ is large enough, each variable needs to significantly reduce the error in order to justify the cost of including it into the model. Otherwise, it is better to set $\beta_i = 0$. 

When $\lambda = 0$, we recover the least squares model, while as $\lambda$ increases, the model selects various subsets of variables. When $\lambda \to \infty$, all regression coefficients become $0$ as it becomes too costly to include variables in the model. 

In theory, the above solves our problem of identifying relevant subsets of variables. However, unfortunately, the above optimization problem is a combinatorial optimization problem that cannot be solved efficiently. We therefore need to look for alternatives. 

### Ridge regression/Tikhonov regularization: 

Ridge regression solves the following problem: 

\begin{align*}
\hat{\beta}^{\textrm{ridge}} &= \textrm{argmin}_{\beta \in \mathbb{R}^p} \|y - X\beta\|_2^2 + \lambda \sum_{i=1}^p \beta_i^2 \\
&= \textrm{argmin}_{\beta \in \mathbb{R}^p} \|y - X\beta\|_2^2 + \lambda \|\beta\|_2^2.
\end{align*}

Notice how the term $\sum_{i=1}^p {\bf 1}_{\beta_i \ne 0}$ that previously counted the number of nonzero regression coefficients was replaced by a measure of "how large" the coefficients are (using the $\ell_2$ norm $\|x\|_2^2 = \sum_{i=1}^p x_i^2$). Thus, the above problem penalizes having large regression coefficients. 

One can show that the Ridge regression problem is equivalent to solving 

$$
&\min_{\beta \in \mathbb{R}^p} \|y - X\beta\|_2^2 \\
&\textrm{such that } \|\beta\|_2 \leq t
$$
for some $t > 0$ that depends on $\lambda$. Ridge regression thus solves the usual regression problem, but looks for a solution in a ball of radius $t$. The solution can easily be written in closed form by using our previous [calculations](sec-finding-optimal-coefficients) for least squares. Indeed, we have

\begin{align*}
\nabla_\beta \left(\|y - X\beta\|_2^2 + \lambda \sum_{i=1}^p \beta_i^2\right) &= 2 (X^TX \beta - X^Ty) + 2 \lambda \beta \\
&= 2\left((X^TX + \lambda I)\beta - X^Ty\right).
\end{align*}

Therefore, the critical points satisfy

$$
(X^T X + \lambda I)\beta = X^T y.
$$


Notice that the matrix $(X^T X + \lambda I)$ is positive definite, and therefore invertible. The unique solution to the Ridge regression problem is therefore

$$
\beta^{\textrm{ridge}} = (X^T X + \lambda I)^{-1} X^T y.
$$

```{admonition} Positive semidefinite and positive definite matrices

A symmetric matrix $A \in \mathbb{R}^{n \times n}$ is said to be positive semidefinite if $x^T A x \geq 0$ for all $x \in \mathbb{R}^n$. Similarly, it is said to be positive definite if $x^T A x > 0$ for all $x \in \mathbb{R}^n \setminus \{\mathbf{0}\}$. Such matrices appear naturally in many areas of sciences and engineering. 

Equivalently, one can show that a matrix is positive semidefinite if and only if all its eigenvalues are nonnegative (and positive definite if and only if its eigenvalues are all positive). 

The matrix $X^TX$ above is positive semidefinite since for any $x \in \mathbb{R}^p$, we have 

$$
x^T (X^TX)x = (Xx)^t(Xx) = \|Xx\|_2^2 \geq 0.
$$

Thus, the eigenvalues of $X^TX$ are non-negative. Adding $\lambda I$ to it adds $\lambda$ to each eigenvalue. Thus $X^TX + \lambda I$ has positive eigenvalues and, in particular, is invertible (since $0$ is not one of its eigenvalues). 

``` 

```{admonition} Ridge regression: a University of Delaware success story! 

Amazingly, Ridge regression was originally developed by Arthur Hoerl and Robert "Bob" Kennard at the University of Delaware. See the following <a href="https://lerner.udel.edu/seeing-opportunity/blue-hens-revolutionize-precursor-to-machine-learning/" target="_blank">article</a> for more details. 
```

The Scikot-learn package has a <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html" target="_blank">Ridge</a> object that can be used to solve the Ridge regression problem: 

```python
from sklearn.linear_model import Ridge
```

The object has similar methods (fit, predict, etc.) as the linear regression object. See the <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html" target="_blank">documentation</a> for more details.

Notice that, compared to least squares, the Ridge regression solution is unique. Adding $\lambda I$ to the $X^TX$ term thus provides a rigorous way to "regularize" the linear regression problem. This is particularly important when the number of samples $n$ is smaller than the number of varibles $p$. In that case, the matrix $X^TX$ is not invertible and the least squares solution is not unique. However, the Ridge regression coefficients are typically not equal to $0$. Instead, as the penalty parameter increases, the coefficients shrink towards $0$ (without typically being exactly zero). Ridge regression is thus used as a way to **regularize** regression and not as a way to select the best predictors to use. As we will see in the next section, another modification of our original problem (called LASSO regression) allows us to select predictors.

To illustrate how Ridge regression behaves, let us try it on a standard dataset. We will use the <a href="https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset" target="_blank">diabetes</a> dataset that comes with scikit-learn. Please make sure that you understand the code below and can run it by yourself.

```python

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

# Load the diabetes dataset
X, y = load_diabetes(return_X_y=True)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Range of alpha (penalty) values
alphas = np.logspace(-4, 4, 100)

# Store coefficients for each alpha
coefs = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_scaled, y)
    coefs.append(ridge.coef_)

# Convert list to NumPy array for plotting
coefs = np.array(coefs)

# Plot the coefficients as a function of the regularization
plt.figure(figsize=(10, 6))
for i in range(coefs.shape[1]):
    plt.plot(alphas, coefs[:, i], label=f'Feature {i}')

plt.xlabel('Alpha (Regularization strength)')
plt.ylabel('Coefficient values')
plt.title('Ridge coefficients as a function of regularization')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

```
We obtain the following figure displying the behavior of the coefficients as a function of the penalty parameter.

```{figure} images/ridge-example.png
---
width: 800 px
---
Behavior of the Ridge regression coefficients as a function of the penalty parameter for the diabetes dataset.
```

Observe how the coefficients shift towards zero as the penalty parameter increases. However, they are typically not equal to $0$ for a given choice of the penalty parameter. 

## LASSO regression

In contrast to Ridge regression, LASSO (Least Absolute Shrinkage and Selection Operator) regression does set coefficients to $0$ and selected relevant subsets of predictors in the linear regression problem. LASSO solves the following problem: 

\begin{align*}
\hat{\beta}^{\textrm{lasso}} &= \textrm{argmin}_{\beta \in \mathbb{R}^p} \left(\|y - X\beta\|_2^2 + \lambda \sum_{i=1}^p |\beta_i|\right) \\
&= \textrm{argmin}_{\beta \in \mathbb{R}^p} \left(\|y - X\beta\|_2^2 + \lambda \|\beta\|_1\right).
\end{align*}

Observe that the penalty term now uses the $\ell_1$ norm instead of the $\ell_2$ norm as in Ridge regression. This small modification has a profound effect on the properties of the solution. As for Ridge regression, one can show that the above problem is equivalent to solving 

\begin{align*}
&\min_{\beta \in \mathbb{R}^p} \|y-X\beta\|_2^2 \\
&\textrm{such that } \|\beta\|_1 \leq t.
\end{align*}

Thus, LASSO regression minimizes the usual error, but looks for a solution in the $\ell_1$ ball of radius $t$. Observe that the shape of balls in the $\ell_1$ norm is different from the shape of the balls in the $\ell_2$ norms. For example, let $v = (x,y) \in \mathbb{R}^2$. The standard $\ell_2$ unit ball is 

$$
\|v\|_2 \leq 1 \iff \sqrt{x^2 + y^2} \leq 1 \iff x^2 + y^2 \leq 1. 
$$

This is the unit disk. However, for the $\ell_1$ norm, we have 

$$
\|v\|_1 \leq 1 \iff |x| + |y| \leq 1. 
$$

Observe that the ball is bounded by the four lines $\pm x \pm y = 1$ (consider four cases, depending on the signs of $x$ and $y$).  

```{figure} images/l1-ball.png
---
width: 500 px
---
The $\ell_1$ unit ball in $\mathbb{R}^2$. 
```

Let us now look at how the solution of the Ridge and the LASSO problem differ. Recall that each problem minimizes the error $\|y-X\beta\|_2^2$ of the linear regression, but over $\ell_2$ and $\ell_1$ balls respectively. We illustrate the difference in $\mathbb{R}^2$. 

```{figure} images/Fig3p11.png
---
width: 500 px
---
ESL, Fig. 3.11.
```

On the figure $\hat{\beta}$ is the least squares solution of the regression problem, i.e., the value of $\beta$ that minimizes $\|y-X\beta\|_2^2$. The red curves are <a href="https://en.wikipedia.org/wiki/Level_set" target="_blank">level curves</a> of $\|y-X\beta\|_2^2$, i.e., on each red curve, the value of the error is the same. Observe that the value of the error on each red curve is larger than the error at $\hat{\beta}$ since $\hat{\beta}$ is where the error is minimized. As we move away from $\hat{\beta}$, the error keeps increasing. The Ridge and LASSO solutions are obtained when a level curve intersects a unit ball (in the $\ell_2$ norm for Ridge and in the $\ell_1$ norm for LASSO). Observe how the shape of the $\ell_1$ ball makes it more likely that the intersection will occur at a "corner" of the ball, i.e., at a location where one of the regression coefficients is equal to $0$. The above argument is far from a formal proof that the LASSO solution has many coefficients equal to $0$, but it provides some intuition to explain what happens. 

Let us repeat our experiment with the diabetes dataset to see how the LASSO coefficients behave. We can again use scikit-learn: 

```python
from sklearn.linear_model import Lasso
```

See the <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html" target="_blank"> documentation</a>.

Let us modify our previous code to use this object instead of the Ridge object. 

```python

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

# Load the diabetes dataset
X, y = load_diabetes(return_X_y=True)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Range of alpha (penalty) values
alphas = np.logspace(-3, 0, 100)

# Store coefficients for each alpha
coefs = []

for alpha in alphas:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_scaled, y)
    coefs.append(lasso.coef_)

# Convert list to NumPy array for plotting
coefs = np.array(coefs)

# Plot the coefficients as a function of the regularization
plt.figure(figsize=(10, 6))
for i in range(coefs.shape[1]):
    plt.plot(alphas, coefs[:, i], label=f'Feature {i}')

plt.xlabel('Alpha (Regularization strength)')
plt.ylabel('Coefficient values')
plt.title('Lasso coefficients as a function of regularization')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

```
We now obtain the following figure:

```{figure} images/lasso-example.png
---
width: 800 px
---
Behavior of the LASSO regression coefficients as a function of the penalty parameter for the diabetes dataset.
```

Observe how each cofficients becomes $0$ once the penalty parameter reaches a certain point. Another surprising feature of the solution is that the coefficients are piecewise linear. This is very different from the Ridge solution! As we keep increasing the penalty parameter, more and more regression coefficients are set to $0$. In practice, one typically solves the LASSO problem for a small number of penalty parameters (say 10 or 100) and looks at the nonzero regression coefficients. These provide relevant candidate subsets of predictors to try instead of trying all possible subsets as in [subset selection](sec-subset-selection). One can then choose the model resulting in the smallest test error. In some problems, it is very interesting to look at the predictors selected by the LASSO. For example, in the [gene expression example](E-gene-expression), it is very interesting to look at the literature and consult with experts to see if the genes selected by the LASSO are known to be related to a given cancer. 

```{note}
Keep in mind that the LASSO is normally used **only to select relevant predictors**. The estimated regression coefficients are not usually directly used. Instead, after using the LASSO to pick relevant variables, one typically fits a standard linear regression model to the subset of predictors. 
```