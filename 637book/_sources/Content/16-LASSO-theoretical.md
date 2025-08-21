# Theoretical guarantees for the LASSO

So far, we saw how the LASSO has the potential to recover the nonzero coefficients $\beta_1, \dots, \beta_p$ if $y = \sum_{i=1}^p \beta_i X_i$ for some predictors $X_1, X_2, \dots, X_p$. Even when the sample size is small, the LASSO may be able to recover the nonzero coefficients if there are not too many of them. We saw that in action in [](S-Lab-LASSO). 

In order for the above process to be used widely, it is important to have some **theoretical guarantees** that the LASSO will indeed recover the nonzero regression coefficients in such a setting. In this chapter, we briefly explain one such theoretical result. We begin by constructing a natural probabilistic framekwork to study the problem. 

## A probabilistic model

In order to study the behavior of the LASSO on a given dataset, we make the following **assumptions**: 

1. $X_1, \dots, X_p$ are (possibly dependent) random variables.
2. $|X_j| \leq M$ almost surely for some $M > 0$, $(j=1,\dots,p)$.
3. $Y = \sum_{j=1}^p \beta_j^* X_j + \epsilon$ for some (unknown) constants $\beta_j^*$. 
4. $\epsilon \sim N(0,\sigma^2)$ is independent of the $X_j$ ($\sigma^2$ unknown).
5. $\sum_{j=1}^p |\beta_j^*| \leq K \textrm{ for some } K > 0 \textrm{ (sparsity assumption)}$.

Notice that Assumption 3 states that $Y$ depends linearly on the predictors, but with some added noise $\epsilon$.  

Next, assume we are given $n$ iid (independent and identically distributed) observations

$$
Z_i = (Y_i, X_{i,1}, \dots, X_{i,p})
$$

of $(Y, X_1, \dots, X_p)$. 

Our goal is to recover $\beta_1^*, \dots, \beta_p^*$ as accurately as possible using only the observed data $Z_i$ for $i=1,\dots,n$.

## Measuring the error

Let 

$$
\hat{Y} = \sum_{j=1}^p \beta_j^* X_j
$$

be the best predictor of $Y$ if the true coefficients were known. Given some estimate of the regression coefficients $\tilde{\beta}_1, \dots, \tilde{\beta}_p$, let 

$$
\tilde{Y} = \sum_{j=1}^p \tilde{\beta}_j X_j
$$

be the associated predictor of $Y$. Define the *mean square prediction error* by

$$
\textrm{MSPE}(\tilde{\beta}) = E(\hat{Y} - \tilde{Y})^2.
$$

In other words, $\textrm{MSPE}(\tilde{\beta})$ measures how well our predictor with estimated regression coefficients does compared to the optimal predictor with the exact coefficients. 

The following theorem provides a bound on $\textrm{MSPE}(\tilde{\beta})$ when $\tilde{\beta}$ is the LASSO solution.

 **Theorem:** (Chatterjee, <a href="https://arxiv.org/abs/1303.5817" target="_blank">Assumptionless consistency of the Lasso</a>, 2014) Under the above assumptions, the LASSO solution $\tilde{\beta}$ satisfies:

$$
\textrm{MSPE}(\tilde{\beta}) \leq 2KM \sigma \sqrt{\frac{2 \log(2p)}{n}} + 8 K^2 M^2 \sqrt{\frac{2\log(2p^2)}{n}}.
$$

The important point here is that $p$ can be large compared to $n$. As long as $\log(p)/n$ is small, the average squared error made by the LASSO will be small. 

```{note}

We conclude this brief chapter by noting that while results such as the above theorem may not be of direct use to the practitioner, they are fundamentally important to guarantee that tools such as the LASSO produce the desired results.   

```