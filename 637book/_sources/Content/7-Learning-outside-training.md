(C-learning-outside)=
# Learning outside the training set

We will now examine in more details the learning problem and its different components. 

```{figure} images/learning_pb.png
---
width: 600 px
---
Source: Abu-Mostafa et al., ``Learning from data''.
```

In a typical learning problem, one is interested to learn a function $f: \mathcal{X} \to \mathcal{Y}$ connecting inputs to outputs. For example, the input could be a grayscale image containing a handwritten digit and the output the corresponding digit. One could imagine there is some optimal function $f$ connecting such inputs to outputs. In order for the machine to approximate that function, several training pairs of inputs/outputs are provided. 

The *hypothesis set* is a set of function that is being considered for $f$. This is often a set of functions parametrized by a finite dimensional vector space. For example, in linear regression, the hypothesis set $\mathcal{H}$ is the set of linear functions.

```{note}
In machine learning, we usually restrict ourselves to a particular hypothesis set as the set of all functions from $\mathcal{X}$ to $\mathcal{Y}$ is usually too big to search. 
```

Once a hypothesis set has been selected, a learning algorithm searches the hypothesis set to find a function $g \in \mathcal{H}$ that best matches the data. We call this function the final hypothesis. 

## Selecting a good hypothesis set

When choosing a hypothesis set, one needs to find a good trade-off between underfitting and overfitting: 

* **Underfitting**: a model that is too simple will fail to capture the complexity of the data. 
* **Overfitting**: a model that is too complex will learn all the small variations in the data and will generalize poorly to new data. 

To help guide this choice and provide a better guarantee that the model will generalize well, we proceed as follows: 

1. We split our sample into 2 parts (training data and test data) as
uniformly as possible. People often use 75% training, 25% test.
2. We fit our model using the training data only. (This minimizes
the **training error**).
3. We use the fitted model to predict values of the test data and
compute the **test error** (This estimates the performance of the model on new data that were not seen during training).

```{figure} images/train-test.jpg
---
width: 500 px
---
```

## Example: least squares
In the case of least squares, the regression coefficients estimated on the training set are given by (assuming the matrix is invertible): 

$$
\hat{\beta} = (X^T_{\textrm{train}} X_{\textrm{train}})^{-1} X_{\textrm{train}}^T Y_{\textrm{train}}.
$$ 

The predicted values on the test set are therefore: 

$$
\widehat{Y}_{\textrm{test}} = X_{\textrm{test}}\hat{\beta}.
$$

Finally, the test error is:  

$$
\textrm{MSE}_{\textrm{test}} = \frac{1}{n_2} \sum_{i=1}^{n_2} (\widehat{Y}_{\textrm{test},i} - Y_{\textrm{test},i})^2.
$$

If we have different competing linear regression models (obtained from, say, different transformations of the variables), **the test error can be used to guide the choice of the final model**. 

## Training and test error

The following figure illustrates the typical behavior of the training and test errors as a function of the complexity of the model. 

```{figure} images/ESL-Fig2p11.png
---
width: 600 px
---
Source: ESL, Figure 2.11.
```

For example, in a linear regression model: 
* As we keep adding more variables, the training error **always** decreases. This is because, when we add variables, the original model is contained in the bigger one (just set the regression coefficients corresponding to the new variables to $0$). 
* However, typically, the test error will start increasing at some point. This is because the model with too many variables becomes very flexible and starts learning irrelevant patterns in the data (the model is **overfitting**).

```{admonition} Splitting a dataset into training and test sets with Python
:name: R-train-test-split
Scikit-learn provides a <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html" target="_blank">train_test_split</a> function to split the data automatically for
us. Try it to measure the test error of the different linear models you previously built for the JSE_Car_Lab dataset!
```python
from sklearn.model_selection import train_test_split
# Split data into training and test sets
X_train, X_test, y_train, y_test =
train_test_split(X, y, test_size=0.25,
random_state=42)
# Fit model on training data
lin_model = LinearRegression()
lin_model.fit(X_train,y_train)
# Returns the coefficient of determination R^2.
lin_model.score(X_test, y_test)
```

(sec-r-squared)=
## The coefficient of determination 

Regression models are often ranked using the *coefficient of determination* called "R squared" and denoted $R^2$. Suppose $y_1, y_2, \dots, y_n$ are the outputs of a dataset and $\widehat{y}_1, \widehat{y}_2, \dots, \widehat{y}_n$ are the associated values predicted by a model. Then the $R^2$ is given by: 

$$
R^2 = 1 - \frac{\sum_{i=1}^n(y_i-\hat{y_i})^2}{\sum_{i=1}^n (y_i-\overline{y})^2}, 
$$

where $\overline{y} := \frac{1}{n} \sum_{i=1}^n y_i$ is the average of the $y_i$'s. Equivalently, the $R^2$ equals 1 minus the ratio of the MSE of the model, divided by the MSE of a model that predicts $\widehat{y}_i = \overline{y}$ for all $i=1,\dots,n$. Observe that $0 \leq R^2 \leq 1$ for any linear regression model (assuming the model has an intercept). This is because the model is guaranteed to have a MSE smaller than a model with only an intercept. 

When the $R^2$ is close to $0$, the linear regression model is not doing much better than a constant prediction. The model becomes better as the $R^2$ approaches $1$. 

```{admonition} Exercise 
Consider the linear regression model $f(x) = \beta_0$ (i.e., a model making a constant prediction as in the calculation of the $R^2$). It is not difficult to show that, given a training dataset 

$$
(\mathbf{x_1}, y_1), (\mathbf{x_2}, y_2), \dots, (\mathbf{x_n}, y_n), 
$$ 

the intercept $\beta_0$ that minimizes the MSE is $\overline{y}$. Indeed, let 

$$
g(\beta_0) = \frac{1}{n} \sum_{i=1}^n (y_i - \beta_0)^2. 
$$

Use the first derivative test to show that $g$ is minimized when $\beta_0 = \overline{y}$. 
```

The *score* method in *scikit-learn* returns the $R^2$ as a measure of how well the model is doing (see {ref}`(Splitting a dataset into training and test sets with Python)<R-train-test-split>`). 

(sec-cross-validation)=
## Cross validation

A refined approach to splitting a dataset into a training and testing set is to use **cross-validation**. In cross-validation (CV), the samples are split into several subsets (called *folds*) that have roughly equal size. The model is fitted to all the folds *except the first one*. The testing error is then computed on the first fold. This process is repeated with the second fold removed, the third fold removed, etc.. At the end of the process, the average testing error can be computed and provides a rough estimate of how the model will perform on new data.  

```{figure} images/cv.png
---
width: 500 px
---
Cross-validation with $5$ folds where the third fold is currently used for testing.
```

The $K$-fold cross-validation process is summarized below. 

1. Split the samples into $K$ equal (or almost equal) parts/folds at random. 
2. FOR $j=1,\dots,K$: 
4. $\ \ \ $ Fit the model on the data with fold $j$ removed. 
5. $\ \ \ $ Test the model on the $j$-th fold $\rightarrow$ This is the $j$-th test error.
6. ENDFOR
7. Compute the average of the test errors obtained for each fold.

The average error computed in the last step is called the *cross-validation error*. As we will see later, the cross-validation error provides a rigorous way to compare models and to pick the best one (in terms of its ability to predict new values well). 

```{note}
A typical choice for $K$ is $10$. However, if the number of samples is small, one may want to pick a smaller value (e.g. 3 or 5) to make sure enough samples are available for training and testing during the cross-validation process. 
```

## The no free lunch theorem

So far, we have only looked at the linear regression model. We will examine many other models in the upcoming chapters. One may wonder if there is a model that is uniformly superior to all others. For example, for learning things around us, we like to think our brain is pretty effective. An interesting theorem proved in 1996 by Wolpert shows that, in some sense, no learning algorithm is superior to all the others. 

```{admonition} No Free Lunch Theorem (Wolpert, 1996) 
Averaged over all possible data-generating distribution, every classification algorithm has the same error rate when classifying previously unobserved points.
```

We are not going to rigorously define what it means to "average over all possible data-generating distribution", but roughly speaking, the theorem says that if you generate data accoding to a very large number of probability distributions, then the average test error of all models is about the same. Hence, given **any** learning algorithm, there is at least one dataset where it will not perform well. There is thus no *universal* learning algorithm that performs well in all situations.

In practice, however, we do not care about doing well on **all** datasets. A more reasonable goal is to understand what kind of data distributions are
relevant to the "real world" or in a certain field of study, and find learning algorithms that perform well on such datasets. For example, one can look for learning algorithms that work well for detecting objects in images. 

In some sense, even our brain is very specialized: it works well for learning tasks involving the objects we regularly interact with (e.g., for distinguishing different animals). However, our brain is not effective for any task. For example, the images below contain 51% and 49% white pixels respectively. Is our brain really effective at learning to differentiate the two types of pictures? Would looking at thousands of images of each kind really help?

```{figure} images/pixels-51.png
---
width: 500 px
---
An image with 51% white pixels
```

```{figure} images/pixels-49.png
---
width: 500 px
---
An image with 49% white pixels
```

## Theoretical guarantees

We now discuss how some theoretical guarantees can be provided that a model will perform well if it is trained on a large random sample. 

As above, let $f: \mathcal{X} \to \mathcal{Y}$ be a target function we are trying to approximate (e.g. how images of hand-written digits are related to the actual digit). Let $\mathcal{D}$ be our training dataset. 

**Question:** Does the training data tells us anything *outside* $\mathcal{D}$? 

### Example: learning a boolean function (Abu-Mostafa et al., Section 1.3.1)

Suppose $\mathcal{X} = \{0,1\}^3$ and $\mathcal{Y} = \{0,1\}$, i.e., we are trying to learn a boolean function on $\{0,1\}^3$.

```{admonition} Exercise
Explain why there are $2^{2^3} = 256$ such functions.
```

Suppose we are given the following training data, where the black dots indicate a value of $0$ and the white dots a value of $1$: 

```{figure} images/Learning_1.3.1.png
---
width: 150 px
---
Source: Abu-Mostafa et al., ``Learning from data''.
``` 

Observe that we know the values of $f$ on $5$ of the $2^3 = 8$ inputs (62.5% of the possible values). Does that tell us anything about the values of $f$ on the remaining three possible inputs? The figure below displays all the functions that match out training data

```{figure} images/Learning_1.3.3_b.png
---
width: 500 px
---
```

Clearly, we are free to choose the values of $f$ on the remaining 3 points of $\mathcal{X}$ arbitrarily. 

### A probabilistic approach

The previous example shows that, in general, there is no hope to learn the unknown function $f$ exactly in a given learning problem. Even if our final hypothesis $g$ does well on the training set, there is no guarantee it will do well outside the training set. However, we can show that with a large enough dataset, we can learn $f$ well enough *outside $\mathcal{D}$* with **high probability**. 

#### Example: sampling marbles

Consider a bin with red and green marbles. Let $\mu$ denote the exact proportion of red marbles in the bin. Assume this number is unknown to us. Naturally, we can sample marbles at random and use the proportion of red marbles in that sample as an estimate for $\mu$. 

```{figure} images/Learning_1.3.2.png
---
width: 500 px
---
Source: Abu-Mostafa et al., ``Learning from data''.
```

```{note}
Notice how the above problem arises when performing a survey to estimate the proportion of people having a given opinion (e.g. in an election poll). 
```

Suppose we sample $N$ marbles with replacement to infer the value of $\mu$. Note that there is no guarantee that we can learn $\mu$ *exactly*. For example, even if $\mu = 0.1$, we could end up picking the same red marble $N$ times and conclude $\mu = 1$. However, the probability of the above happening is ridiculously small. We *can* learn $\mu$ *with high probability* if we sample enough marbles.

The following inequality provides a bound on how good our estimate of $\mu$ is as a function of the number of marbles $N$ we sampled. 

```{admonition} **Theorem:** (Hoeffding's Inequality)

Let $X_1, X_2, \dots, X_N$ be independent random variables such that $a_i \leq X_i \leq b_i$ almost surely. Consider the sum 

$$
S_N = X_1 + X_2 + \dots + X_N.
$$

Then, for any $\epsilon > 0$, 

$$
P\left(|S_N - E(S_N)| \geq \epsilon \right) \leq 2 \exp\left(-\frac{2 \epsilon^2}{\sum_{i=1}^N (b_i-a_i)^2}\right).
$$
```

Let us see how we can apply Hoeffding's Inequality to our problem. Let

$$
X_i = \begin{cases}
1 & \textrm{if the } i\textrm{-th marble is red} \\
0 & \textrm{otherwise}.
\end{cases}
$$

We have: 
* $S_N = $ number of red marbles picked (random)
* $\nu := S_N / N = $ estimated value of $\mu$ (random). 

Observe that: 
* $E(X_i) = 1 \times \mu + 0 \times (1-\mu) = \mu$ and so $E(\nu) = E(S_N/N) = \mu$. 
* It is natural to assume $X_1, X_2, \dots, X_N$ are independent. 
* We have $0 \leq X_i \leq 1$. 

Thus, by Hoeffding's inequality, for any $\epsilon > 0$: 

$$
P(|\nu - \mu| > \epsilon) = P(|S_N - E(S_N)| \geq \epsilon N) \leq 2\exp\left(-2 \frac{\epsilon^2 N^2}{N}\right) = 2 e^{-2 \epsilon^2 N}. 
$$ 

In particular, for any $\epsilon > 0$, we have 

$$
P(|\nu - \mu| > \epsilon) \to 0 \textrm{ as } N \to \infty. 
$$

Thus, for any choice of error $\epsilon > 0$, the probability tha the estimated value $\nu$ differs from $\mu$ by more than $\epsilon$ becomes very small as the number of samples $N$ increases. 

We can say even more. Suppose we want to make sure our estimate is accurate at $\pm 5\%$ with a $95\%$ probability. Can we pick the value of $N$ to guarantee that? In that case, we have 

$$
P(|\nu - \mu| > 0.05) \leq 2 e^{-2 \times 0.05^2 N}. 
$$

Hence, we want to pick $N$ large enough so that $2 e^{-2 \times 0.05^2 N} \geq 0.95$. Solving for $N$, we obtain $N \geq  149$. We can thus guarantee that our estimate is accurate at $\pm 5\%$ with probability $95\%$ if $N \geq 149$.

In general, observe that if $\epsilon$ is very small (the estimate is very precise), the sample size $N$ needs to be larger to guarantee accuracy of the estimator. 

```{note}
Suppose for example that we want to survey the University's population to know if people are in favor of a given project (say, having more online courses). The above calculation shows that if we survey at least $149$ people (as randomly as possible), the estimated proportion of people in favor will be accurate at $\pm 5\%$ with a $95\%$ probability.  
```

#### Back to predicting outside the training set

We will now show how the above arguments can be applied to the learning problem. Assume in our learning problem that the data is given to us at random according to some unknown probability distribution. Also assume we already have a training dataset of size $N$. We want to know if a model trainined on the training set is guaranteed to perform well on the test set. 

We will use the following notation: 

$$
[[\textrm{statement}]] = \begin{cases}
1 & \textrm{ if statement is true and }\\
0 & \textrm{ otherwise}.
\end{cases}
$$

We define the: 
* In-sample error (training error):


\begin{align*}
E_{\textrm{in}}(g) &= (\textrm{fraction of $\mathcal{D}$ where $f$ and $g$ disagree}) \\
&= \frac{1}{N} \sum_{n=1}^N [[g({\bf x_n}) \ne f({\bf x_n})]]
\end{align*}


* Out-of-sample error: 

$$
E_{\textrm{out}}(g) = P(g({\bf x}) \ne f({\bf x})), 
$$

where ${\bf x}$ is sampled according to the unknown data distribution.

Our goal is to: 
* Find an hypothesis $g$ for which $E_{\textrm{in}}(g)$ is small (good training error). 
* Prove that $E_{\textrm{out}}(g)$ and $E_{\textrm{in}}(g)$ are not too different with high probability.
* Conclude that $E_{\textrm{out}}(g)$ is small with high probability.

We do this by relating the problem to the problem of sampling marbles described above. 

To simplify, let us assume there are only a finite number of hypotheses:

$$
\mathcal{H} = \{h_1, h_2, \dots, h_M\}.
$$

(Recall that the hypothesis set is the set of functions we are considering to approximate $f$.) 

Let us pick the hypothesis that does the best on the training set (i.e., $E_{\textrm{in}}(g)$ is small). We would like to guarantee that $E_{\textrm{out}}(g)$ is small as well. 

Each $h_i$ disagrees with $f$ at certain points ${\bf x_i}$ (red marble) and agrees with $f$ at certain points (green marble):

```{figure} images/Fig-1.10.png
---
width: 500 px
---
```

Observe how our training set provides a random sample of marbles for which we know the proportion of red balls. Hoeffding's inequality holds in each bin *individually*: 

$$
P(|E_{\textrm{in}}(h_i) - E_{\textrm{out}}(h_i)| > \epsilon) \leq 2e^{-2\epsilon^2 N} \qquad (i=1,\dots, M)
$$

and guarantees that if a model does well on the training set (proportion of red marbles is small), then it will do well on the test set (assuming $N$ is large enough).

There is just one more subtlety that we need to address before concluding out model will do well on the test set: the above bound assumes $h_i$ is fixed before the sample is seen. However, in practice, we want to pick the "best" $h_i$ (the one with the smallest in-sample error). We therefore decide which $h_i$ to pick *after* seeing the data so the above bound does not immediately apply. Right now, we know: 

$$
P(|E_{\textrm{in}}(h_i) - E_{\textrm{out}}(h_i)| > \epsilon) \textrm{ is small for any particular } h_i.
$$

What we want is: 

$$
P(|E_{\textrm{in}}(g) - E_{\textrm{out}}(g)| > \epsilon) \textrm{ is small for the final hypothesis } g, 
$$

where $g$ is the $h_i$ with the smallest training error. We can easily fix the problem using a *union bound* 

```{admonition} The union bound in probability theory: 
Suppose $E_1, E_2, \dots, E_M$ are events. Then the probability of their union (at least one of the events happens) is bounded by 

$$
P(E_1 \cup E_2 \cup \dots \cup E_M) \leq \sum_{i=1}^M P(E_i). 
$$
```

We apply the union bound as follows: 

If

$$
|E_{\textrm{in}}(g) - E_{\textrm{out}}(g)| > \epsilon
$$

then we must have

\begin{align*}
&\ \  |E_{\textrm{in}}(h_1) - E_{\textrm{out}}(h_1)| > \epsilon \\
&\textrm{or } |E_{\textrm{in}}(h_2) - E_{\textrm{out}}(h_2)| > \epsilon \\
& \dots \\
&\textrm{or } |E_{\textrm{in}}(h_M) - E_{\textrm{out}}(h_M)| > \epsilon.
\end{align*}



Let $\mathcal{B}_i$ be the event: 

$$
\mathcal{B}_i := |E_{\textrm{in}}(h_i) - E_{\textrm{out}}(h_i)| > \epsilon.
$$

Then


\begin{align*}
&P(|E_{\textrm{in}}(g) - E_{\textrm{out}}(g)| > \epsilon) \leq P(\mathcal{B}_1 \cup \mathcal{B}_2 \cup \dots \cup \mathcal{B}_M) \\
&\leq \sum_{i=1}^M P(\mathcal{B}_i) = \sum_{i=1}^M P(|E_{\textrm{in}}(h_i) - E_{\textrm{out}}(h_i)| > \epsilon) \\
&\leq \sum_{i=1}^M 2e^{-2\epsilon^2 N} \\
&= 2M e^{-2\epsilon^2 N}.
\end{align*}


**Conclusion:** We get $P(|E_{\textrm{in}}(g) - E_{\textrm{out}}(g)| > \epsilon) \to 0$ as $N \to \infty$ (with an explicit control on the error). 

The downside of the above argument is it requires $M$ to be finite (and we get a looser bound). This can be improved with more work. 

The above calculations show that (under the assumptions we made), we are guaranteed that a chosen model that performs well on a large enough training set will also perform well on new data with high probability. This provides a **theoretical guarantee** that learning the function $f$ is possible (with high probability). We note, however, the the kind of bounds we obtained are typically not very useful in practice in the learning problem (for example, to guide the choice of the training set size, as we did above for the pooling problem). 
