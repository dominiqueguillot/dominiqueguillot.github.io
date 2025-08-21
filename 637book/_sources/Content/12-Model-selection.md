# Model selection

As we saw, both Ridge and LASSO regression depend on a parameter $\lambda > 0$. In general, models for data can depend on millions or even billions of parameters. [Cross-validation](sec-cross-validation) is typically used in the training phase to select parameters. 

Let $f_\lambda$ be some model that depends on a parameter $\lambda$. For example, $f_\lambda$ could be Ridge regression with penalty $\lambda$. Assume we are given training data $(y_1, \mathbf{x_1}), (y_2, \mathbf{x_2}), \dots (y_n, \mathbf{x_n})$. Our goal is to choose $\lambda$ in a good way so that $f_\lambda(\mathbf{x_i}) \approx y_i$ for all $i$. Let $L(y,\hat{y})$ be a loss function to measure the error made by the model. For example, if our prediction is a one-dimensional real number, we could pick $L(y,y_i) = |y-y_i|^2$. One possible option to choose $\lambda$ would be to pick the parameter that minimizes the average error, i.e., 

$$
\widehat{\lambda} = \mathop{\textrm{argmin}}_\lambda \frac{1}{n} \sum_{i=1}^n L(y_i, f_\lambda(\mathbf{x_i})) = \mathop{\textrm{argmin}}_\lambda \frac{1}{n} \sum_{i=1}^n |y_i - f_\lambda(\mathbf{x_i})|^2. 
$$

The problem with that approach though is that the model learns to predict the training data well, but may not generalize well to new data. Instead, we use a cross-validation approach to pick the parameter $\lambda$. 

Recall that in $K$-folds cross-validation, the training data is split into $K$ folds $F_1, \dots, F_K$. 

```{figure} images/cv.png
---
width: 500 px
---
```

Let $f_\lambda^{-k}({\bf x})$ be the model fitted on the training data, excluding the $k$-th fold. We define the *cross-validation error* to be the average error made by the model on folds $F_1, F_2, \dots, F_K$: 

$$
\textrm{CV}(\lambda) := \frac{1}{K} \sum_{k=1}^K  \frac{1}{|F_k|}\sum_{i \in F_k} L(y_i, f_\lambda^{-i}({\bf x}_i))
$$

The cross-validation error much better measures how the model with parameter $\lambda$ generalizes to new data. We therefore pick $\lambda$ to minimize that error: 

$$
\widehat{\lambda}_\textrm{CV} = \mathop{\textrm{argmin}}_\lambda\ \textrm{CV}(\lambda).
$$

In practice, it can be very difficult to minimize the cross-validation error over all possible values of $\lambda$. Instead, one typically picks a finite number of relevant parameter values, say $\lambda_1, \dots, \lambda_N$, and picks the model with the smallest cross-validation error: 

$$
\widetilde{\lambda}_\textrm{CV} = \mathop{\textrm{argmin}}_{\lambda \in \{\lambda_1, \dots, \lambda_N\}}\ \textrm{CV}(\lambda).
$$