# Best linear unbiased estimator and the bias-variance decomposition

## The Gauss-Markov theorem

As we saw in [](sec-finding-optimal-coefficients), for $X \in \mathbb{R}^{n \times p}$ and $y \in \mathbb{R}^n$, the optimal regression coefficients $\widehat{\beta}$ for predicting $y$ using the columns of $X$ are given by:

$$
\widehat{\beta}_\textrm{LS} = (X^T X)^{-1} X^T y. 
$$

They are optimal in the sense that they minimize the mean squared error (MSE) $\frac{1}{n}\|y-X\beta\|_2^2$.

As we briefly explain in this chapter, $\widehat{\beta}$ has another optimality property: it is the Best Linear Unbiased Estimator (BLUE) for the coefficients. This result is known as the Gauss-Markov theorem. 

```{admonition} Review from probability theory
Let $X, Y$ be random variables. Recall that
1. $E(X)$ is the *expected value* (or mean value) of $X$. 
2. $\textrm{Var}(X) = E((X-E(X))^2)$ is the *variance* of $X$. It measures how far is $X$ from its average, on average. 
3. $\textrm{Cov}(X,Y) = E((X-E(X))(Y-E(Y)))$ is the *covariance* between $X$ and $Y$. Covariance can be thought of as a measure of *linear association*. For example, if $X$ and $Y$ tend to take values greater than their means at the same time, and values smaller than their means at the same time, then $\textrm{Cov}(X,Y) > 0$. 

```{figure} images/covariance.png
---
width: 500 px
---
```

Observe that $\textrm{Var}(X) = \textrm{Cov}(X,X)$. Also, one can show that when $X$ and $Y$ are *independent*, then $\textrm{Cov}(X,Y) = 0$. The converse if false in general. When $\textrm{Cov}(X, Y) = 0$, we say $X$ and $Y$ are *uncorreleated*. This is weaker than assuming $X$ and $Y$ are independent. 

<hr style="border: none; height: 2px; background-color: black;">

We make the following assumptions on our data: ${\bf Y} = {\bf X} \beta + {\bf \epsilon}$, where ${\bf \epsilon} \in \mathbb{R}^n$ with:
1. $E(\epsilon_i) = 0$.
2. $\textrm{Var}(\epsilon_i) = \sigma^2 < \infty$.
3. $\textrm{Cov}(\epsilon_i, \epsilon_j) = 0$ for all $i \ne j$.

These can be summarized by saying that $Y$ really has a linear relationship with $X$, but with some "noise" $\epsilon$ added. The noise has mean $0$ and variance $\sigma^2$, and the noise between samples is uncorrelated (weakly independent). We need two more definitions to state the Gauss-Markov theorem.

A *linear estimator* of $\beta$, is an estimator of the form $\widehat{\beta} = C {\bf Y}$, where $C = (c_{ij}) \in \mathbb{R}^{p \times n}$ is a matrix, and 

$$
c_{ij} = c_{ij}(\bf{X}).
$$

Thus, $\widehat{\beta}$ is a linear combination of $\mathbf{Y}$, where the coefficients can depend on $\mathbf{X}$ (in a possibly non-linear way). Notice that our estimator $\widehat{\beta}_{\textrm{LS}} = ({\bf X}^T {\bf X})^{-1} {\bf X}^T {\bf Y}$ is a linear estimator with $C = ({\bf X}^T {\bf X})^{-1} {\bf X}^T$.

Finally, we say that an estimator is *unbiased* if $E(\widehat{\beta}) = \beta$, i.e., on average, $\widehat{\beta}$ returns the correct value of $\beta$. 

```{admonition} Theorem: (Gauss-Markov)
Suppose ${\bf Y} = {\bf X} \beta + \epsilon$ where $\epsilon$ satisfies the previous assumptions. Let $\widehat{\beta} = C {\bf Y}$ be a linear unbiased estimator of $\beta$. Then for all $a \in \mathbb{R}^p$, 

$$
\textrm{MSE}(a^T \widehat{\beta}_{\textrm{LS}}) \leq \textrm{MSE}(a^T \widehat{\beta}), 
$$

where 

$$
\textrm{MSE}(a^T \widehat{\beta}) = E\left[\left(\sum_{i=1}^n a_i (\widehat{\beta}_i - \beta_i)\right)^2\right] \qquad (a \in \mathbb{R}^p).
$$
```

Intuitively, the theorem says that, under our assumptions, the least squares estimator yields smaller mean square error than any other linear unbiased estimator of $\beta$. The least squares estimator thus has strong theoretical properties. The assumption that $\widehat{\beta}$ is unbiased is very natural (on average, the estimator is correct). However, as we will see in future chapters, one can sometimes get smaller error with working with biased estimators. 

(S-bias-variance)=
## The bias-variance decomposition

Recall that in the Gauss-Markov theorem, we only examined *unbiased* estimators. We will now show that the error of an estimator can be decomposed as a sum of two "types" of error (bias-squared and variance). This is known as the *bias-variance decomposition*. 

Let $\widehat{Z}$ be a random variable trying to predict the value of $z$ (non-random). Then:
* The *bias* of $\widehat{Z}$ is defined by $\textrm{bias}(\widehat{Z}) = z - E(\widehat{Z})$ and measures the difference between the true value $z$, and the average value predicted by $\widehat{Z}$. 
* The *variance* $\textrm{Var}(\widehat{Z})$ measures how much $\widehat{Z}$ varies around its mean. 
* The $\textrm{MSE}(\widehat{Z}-z) = E(\widehat{(Z}-z)^2)$ measures that average squared error made by $\widehat{Z}$ in estimating $z$. 

```{admonition} Theorem: (Bias-variance decomposition)

We have 

$$
\textrm{MSE}(\widehat{Z}-z) = \textrm{bias}(\widehat{Z})^2 + \textrm{Var}(\widehat{Z}).
$$
```
```{dropdown} Proof

We have 

\begin{align*}
\textrm{MSE}(\widehat{Z}-z) &= E(\widehat{(Z}-z)^2) = E(z^2 - 2 z \hat{Z} + \hat{Z}^2) \\
&= E(z^2) - 2 E(z \hat{Z}) + E(\hat{Z}^2) \\
&= z^2 - 2 z E(\hat{Z}) + \textrm{Var}(\hat{Z}) + E(\hat{Z})^2 \\
&= \underbrace{(z-E(\hat{Z}))^2}_{\textrm{bias}^2} + \underbrace{\textrm{Var}(\hat{Z})}_{\textrm{variance}}.
\end{align*}

```

Thus, the theorem indicates two sources of error for an estimator: its bias and its variance. The figure below illustrates this idea. 

```{figure} images/bias-variance.png
---
width: 400 px
---
```

In the figure, the red dot represents the target value $z$. Ideally, we would want an estimator with low bias and low variance (top left). Such estimators have the smallest error. Observe, however, that an estimator with small bias and low variance may be preferable to an estimator with no bias but large variance. One can thus sometimes make a *trade-off* between bias and variance to obtain an estimator with lower error (MSE).  