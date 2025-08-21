# Computing the LASSO solution

The LASSO if often used in high-dimensional problems. As we saw, cross-validation involves solving many lasso problems so it is important to be able to solve the LASSO problem effectively. We will now discuss possible strategies to compute the solution. Recall that the LASSO problem is:

$$
\widehat{\beta}_\textrm{LASSO} = \mathop{\textrm{argmin}}_{\beta \in \mathbb{R}^p} \|y - X\beta\|_2^2 + \alpha \|\beta\|_1.
$$

Notice that the objective function to optimize is not differentiable. As a result, a simple gradient descent cannot be used. 

(S-coordinate-descent)=
## Coordinate descent

A popular approach that is often used to compute the LASSO solution is to use coordinate descent. 

**Objective:** Minimize a function $f: \mathbb{R}^p \to \mathbb{R}$. 

**Strategy:** Starting with an initial guess $x^{(0)} = (x_1^{(0)}, x_2^{(0)}, \dots, x_p^{(0)})$, for $k=0,1,\dots$, minimize each coordinate separately while cycling through the coordinates.

\begin{align*}
x^{(k+1)}_1 &= \mathop{\textrm{argmin}}_x f(x, x_2^{(k)}, x_3^{(k)}, \dots, x_p^{(k)}) \\
x^{(k+1)}_2 &= \mathop{\textrm{argmin}}_x f(x_1^{(k+1)}, x, x_3^{(k)}, \dots, x_p^{(k)}) \\
x^{(k+1)}_3 &= \mathop{\textrm{argmin}}_x f(x_1^{(k+1)}, x_2^{(k+1)}, x, x_4^{(k)}, \dots, x_p^{(k)})\\
&\vdots \\
x^{(k+1)}_p &= \mathop{\textrm{argmin}}_x f(x_1^{(k+1)}, x_2^{(k+1)}, \dots , x_{p-1}^{(k+1)}, x).
\end{align*}

This approach can be very efficient when the coordinate-wise problems are easy to solve (e.g. if they admit a closed-form solution). As we will see, this is the case for the LASSO problem. 


```{figure} images/Coordinate_descent.png
---
width: 500px
---
An illustration of the coordinate descent process. Source: Wikipedia (Nicoguaro)
```

Unfortunately, coordinate descent does **not** always converge. For example, in the example below, observe that if one start with the initial guess $(-2,-2)$, then coordinate descent immediately stops since the function is minimized at that point with respect to the $x$ and $y$ axis directions. However, $(-2,-2)$ is not a local minimum of the function. 

```{figure} images/Nonsmooth_coordinate_descent.png
---
width: 500 px
---
An example where coordinate descent does not converge to a local minimum. Source: Wikipedia (Nicoguaro)
```

However, under supplementary assumptions, one can show that coordinate descent converges to a local minimum of the objective function. To explain the context where it works, we first briefly discuss convex functions.

## Digression: convex functions

### Convex sets and functions

A set $\Omega \subseteq \mathbb{R}^n$ is **convex** if for any $x, y \in \Omega$ and any $0 \leq t \leq 1$, 

$$
tx + (1-t)y \in \Omega.
$$

Equivalently, given any two points in $\Omega$, the line joining them is entirely contained in $\Omega$.

```{figure} images/convex-set.png
---
width: 500 px
---
```

Let $\Omega \subseteq \mathbb{R}^n$ be convex. A function $f: \Omega \to \mathbb{R}$ is said to be **convex** if for all $x, y \in \Omega$ and all $0 \leq t \leq 1$, 

$$
f(t x + (1-t) y) \leq t \cdot f(x) + (1-t) \cdot f(y).
$$

In other words, the function is **under** the line joining any two points (the function "bends up"). 

```{figure} images/Fconvex.png
---
width: 600 px
---
```

**Examples.** Some examples of convex functions include $f(x) = x^2$, $f(x) = e^x$, $f(x) = |x|$, $f(x) = \| y - Ax\|_2^2$. 

### Characterizations of convex functions

Convex functions can be recognized in several ways. 

**Theorem.** A function $f: \Omega \to \mathbb{R}$ is convex if and only if its *epigraph*

$$
\textrm{epi}(f) := \{(x, y) \in \Omega \times \mathbb{R} : y \geq f(x)\}
$$
is a convex set.

**Theorem.** A differentiable function $f$ is convex if and only if 

$$
f(x) \geq f(y) + \nabla f(y)^T (x-y) \qquad \forall x, y \in \Omega.
$$

**Theorem.** A twice differentiable function $f$ is convex if and only if its Hessian matrix $\displaystyle \left(\frac{\partial^2 f}{\partial x_i \partial x_j}\right)$ is positive semidefinite on $\Omega$. 

In particular, the following criterion is very useful for recognizing convex smooth functions in one variable.

**Corollary.** If $f: (a,b) \to \mathbb{R}$ is twice differentiable, then 

$$
f \textrm{ is convex} \iff f''(x) \geq 0 \qquad \forall x \in (a,b).
$$

Some important operations that preserve convexity: 

1. If $f_1, \dots, f_n$ are convex and $w_i \geq 0$, then $\sum_{i=1}^n w_i f_i$ is convex. 
2. In particular, sums of convex functions are convex. 
3. If $f: \mathbb{R}^n \to \mathbb{R}$ is convex and $g: \mathbb{R} \to \mathbb{R}$ is convex and non-decreasing, then $g(f(x))$ is convex. 
4. If $f_1, \dots, f_n$ are convex, then $\max_{i=1}^n f_i$ is convex. 

**Reference:** For more details about convex sets and functions, please see Boyd & Vandenberghe. <a href="https://stanford.edu/~boyd/cvxbook/" target="_blank">Convex optimization</a>, Cambridge university press, 2004 (available for free).

### Sufficient condition for the convergence of coordinate descent

The following result provides a sufficient condition for the convergence of coordinate descent to a local minimum of the objective function.

**Theorem.** (See Tseng, 2001). Suppose

$$
f(x_1, \dots, x_p) = f_0(x_1, \dots, x_p) + \sum_{i=1}^p f_i(x_i) \qquad (f \in \mathbb{R}^p)
$$
satisfies

1. $f_0: \mathbb{R}^p \to \mathbb{R}$ is convex and continuously differentiable.
2. $f_i: \mathbb{R} \to \mathbb{R}$ is convex $(i=1,\dots,p)$.
3. The set $X^0 := \{x \in \mathbb{R}^p : f(x) \leq f(x^0)\}$ is compact. 
4. $f$ is continuous on $X^0$.

Then every limit point of the sequence $(x^{(k)})_{k \geq 1}$ generated by cyclic coordinate descent converges to a global minimum of $f$.

Observe that the theorem makes the assumption that the non-differentiable part of the objective is *separable*, i.e., it can be written as a sum of functions that depend only on one of the variables. This is the case of the LASSO objective (take $f_i(x) = |x_i|$).

As a consequence of the theorem, we obtain that coordinate descent converges to a global minimum for the LASSO problem.

**Corollary.** For the LASSO problem, the coordinate descent iterations converge to a global minimum of the objective function.

It remains to see how the coordinate-wise LASSO problem can be efficiently solved. 

### The coordinate-wise LASSO problem

In order to be able to minimize the LASSO objective using coordinate descent, we need to be able to minimize the objective efficiently each coordinate at a time. Recall that the LASSO objecive is: 

$$
\|y - X\beta\|_2^2 + \lambda \|\beta\|_1. 
$$

Choose $1 \leq i \leq p$. Fix $x_j$ for all $j \ne i$. We want to optimize the differentiable part of the objective over $x_i$:

\begin{align*}
&\min_{x_i} \|y - X\beta\|_2^2 + \lambda \sum_{k=1}^p |\beta_k| \\ 
&=\min_{x_i} \sum_{l=1}^n \left(y_l - \sum_{m=1}^p X_{lm} \beta_m\right)^2 + \lambda \sum_{k=1}^p |\beta_k|.
\end{align*}

We first address the case of the differentiable part $\|y-X\beta\|_2^2$. 

#### Minimizing the differentiable part

Let us first evaluate the gradient of the differentiable part of the objective:

\begin{align*}
\frac{\partial}{\partial x_i}  \sum_{l=1}^n \left(y_l - \sum_{m=1}^p X_{lm} \beta_m\right)^2 &=  \sum_{l=1}^n 2 \left(y_l - \sum_{m=1}^p X_{lm} \beta_m\right) \times (-X_{li}) \\
&= 2 X_i^T (X \beta-y) \\
&= 2 X_i^T (X_{-i} \beta_{-i}-y) + 2 X_i^T X_i \beta_i.
\end{align*}

Here, $X_i$ denotes the $i$-th column of $X$ and $X_{-i}$ is the matrix obtained by deleting the $i$-th column of $X$. 

In order to deal with the non-differentiable part, we need to briefly discuss subdifferential calculus. 

#### Digression: subdifferential calculus

Suppose $f$ is convex and differentiable. Then 

$$
f(y) \geq f(x) + \nabla f(x)^T (y-x).
$$

```{figure} images/Boyd_Fig3p2.png
---
width: 500 px
---
Boyd & Vandenberghe, Figure 3.2.
```

We say that $g$ is a **subgradient** of $f$ at $x$ if 

$$
f(y) \geq f(x) + g^T (y-x) \qquad \forall y.
$$

```{figure} images/Boyd_subgrad.png
---
width: 700 px
---
Boyd, lecture notes.
```

We define

$$
\partial f(x) := \{\textrm{all subgradients of } f \textrm{ at } x\}.
$$

The subgradient generalizes the usual gradient (derivative) for convex functions that are not necessarily differentiable.

1. $\partial f(x)$ is a closed convex set (can be empty). 
2. $\partial f(x) = \{\nabla f(x)\}$ if $f$ is differentiable at $x$.
3. If $\partial f(x) = \{g\}$, then $f$ is differentiable at $x$ and $\nabla f(x) = g$.

In particular, when $f$ is differentiable at $x$, observe that the subgradient $\partial f(x)$ is the set $\{\nabla f(x)\}$ that contains only the gradient of $f$ at $x$. However, when $f$ is not differentiable at $x$, its subgradient can contain more than one point. 

The subgradient behaves as expected with respect to rescaling and adding functions: 

1. $\partial (\alpha f) = \alpha \partial f$ if $\alpha > 0$.
2. $\partial(f_1 + f_2) = \partial f_1 + \partial f_2$.

**Example:** Let us compute the subgradient of the absolute value function.

```{figure} images/abs.png
---
width: 300 px
---
```

When $x \ne 0$, the function is differentiable and so the subgradient contains only the value of the derivative. At $x=0$ however, any slope between $-1$ and $1$ yields a line that remains under the curve. We therefore have:

$$
\partial f(x) = \begin{cases} \{-1\} & \textrm{ if } x < 0 \\ [-1,1] & \textrm{ if } x = 0 \\ \{1\} & \textrm{ if } x > 0. \end{cases}
$$

Now, derivatives provide a very powerful tools to locate the minima of a function. Indeed, recall that if $f$ is convex and differentiable, then

$$
f(x^\star) = \min_x f(x) \Leftrightarrow 0 = \nabla f(x^\star).
$$

The subdifferential calculus allows us to use a similar approach to location the minimum of convex (not necessarily differentiable) functions. 

**Theorem.** Let $f$ be a (not necessarily differentiable) convex function. Then 

$$
f(x^\star) = \inf_x f(x) \Leftrightarrow 0 \in \partial f(x^\star).
$$

*Proof.* We have $f(y) \geq f(x^\star)$ for all $y$ if and only if

$$
f(y) \geq f(x^\star) + 0 \cdot (y-x^\star).
$$

This is equivalent to saying $0 \in \partial f(x^\star)$. ∎

Despite its simplicity, this is a very powerful and important result. We will now use it to compute the coordinate-wise solution of the LASSO problem.

(S-LASSO-soln)=
#### Back to the LASSO solution

We can now complete our calculation to identify the coordinate-wise LASSO solution. 

The function 

$$
f(x_i) := \|y - X\beta\|_2^2 + \alpha \sum_{k=1}^p |\beta_k|
$$
is convex. Its minimum is obtained if $0 \in \partial f(x^\star)$.

Let 

$$
g := \frac{\partial}{\partial x_i} \|y - X\beta\|_2^2=  2\left(X_i^T (X_{-i} \beta_{-i}-y) + X_i^T X_i \beta_i\right).
$$

Then,

$$
\partial f(x_i) = \begin{cases}
\{g- \lambda\} & \textrm{ if } \beta_i < 0 \\
[g -\lambda, g + \lambda] & \textrm{ if } \beta_i = 0 \\
\{g + \lambda\} & \textrm{ if } \beta_i > 0
\end{cases}.
$$

We will find out when each of the above conditions hold. 

First, 

$$
g-\lambda = 0 \Leftrightarrow \beta_i = \frac{2 X_i^T (y-X_{-i} \beta_{-i}) + \lambda}{X_i^T X_i} = g^\star + \frac{\lambda}{\|X_i\|_2^2}.
$$

We therefore conclude that if $\beta_i = g^\star + \frac{\lambda}{\|X_i\|_2^2} < 0$, i.e., if $g^\star < -\frac{\lambda}{\|X_i\|_2^2}$, then $0 \in \partial f(\beta_i)$. 

Similarly, 

$$
g+\lambda = 0 \Leftrightarrow \beta_i = \frac{2 X_i^T (y-X_{-i} \beta_{-i}) - \lambda}{X_i^T X_i} = g^\star - \frac{\lambda}{\|X_i\|_2^2}.
$$

Therefore, if $\beta_i = g^\star - \frac{\lambda}{\|X_i\|_2^2} > 0$, i.e., if $g^\star > \frac{\lambda}{\|X_i\|_2^2}$, then $0 \in \partial f(\beta_i)$. 

What have thus proved so far: 

1. There exists $\beta_i < 0$ such that $0 \in \{g-\lambda\}$ if and only if $g^\star < -\frac{\lambda}{\|X_i\|_2^2}$. In that case, the unique such $\beta_i$ is 

$$
\beta_i = g^\star + \frac{\lambda}{\|X_i\|_2^2}.
$$

2. There exists $\beta_i > 0$ such that $0 \in \{g+\lambda\}$ if and only if $g^\star > \frac{\lambda}{\|X_i\|_2^2}$. In that case, the unique such $\beta_i$ is 

$$
\beta_i = g^\star - \frac{\lambda}{\|X_i\|_2^2}.
$$ 

The remaining case is: $\beta_i = 0$ and $0 \in [g-\lambda, g+\lambda]$.

Recall that

$$
g = 2\left(X_i^T (X_{-i} \beta_{-i}-y) + X_i^T X_i \beta_i\right).
$$

Setting $\beta_i = 0$, we obtain:

\begin{align*}
0 \in [g-\lambda, g+\lambda] &\iff  g-\lambda \leq 0 \textrm{ and } g+\lambda \geq 0 \\
&\iff -\lambda \leq 2X_i^T (y-X_{-i} \beta_{-i}) \leq \lambda \\
&\iff  -\frac{\lambda}{\|X_i\|_2^2} \leq \frac{2 X_i^T (y-X_{-i} \beta_{-i})}{X_i^T X_i} \leq \frac{\lambda}{\|X_i\|_2^2} \\
&= -\frac{\lambda}{\|X_i\|_2^2} \leq g^\star \leq \frac{\lambda}{\|X_i\|_2^2}.
\end{align*}

**Conclusion:** $\beta_i = 0$ and $0 \in [g-\lambda, g+\lambda]$ hold precisely when 

$$
 -\frac{\lambda}{\|X_i\|_2^2} \leq g^\star \leq \frac{\lambda}{\|X_i\|_2^2}.
$$


We have shown the following: 

$$
0 \in \partial f(\beta_i) \textrm{ if } \begin{cases}
\beta_i = g^\star + \frac{\lambda}{\|X_i\|_2^2} &\textrm{ and } g^\star < -\frac{\lambda}{\|X_i\|_2^2} \\
\beta_i = g^\star - \frac{\lambda}{\|X_i\|_2^2} &\textrm{ and } g^\star > \frac{\lambda}{\|X_i\|_2^2} \\
\beta_i = 0 &\textrm{ and } -\frac{\lambda}{\|X_i\|_2^2} \leq g^\star \leq \frac{\lambda}{\|X_i\|_2^2}.
\end{cases}
$$

Therefore, the minimum of $f(\beta_i)$ is obtained at 

$$
x^\star = \begin{cases}
g^\star + \frac{\lambda}{\|X_i\|_2^2} & \textrm{ if } g^\star < -\frac{\lambda}{\|X_i\|_2^2} \\
g^\star - \frac{\lambda}{\|X_i\|_2^2} & \textrm{ if } g^\star > \frac{\lambda}{\|X_i\|_2^2} \\
0 & \textrm{ if }  -\frac{\lambda}{\|X_i\|_2^2} \leq g^\star \leq \frac{\lambda}{\|X_i\|_2^2}.
\end{cases}
$$

In other words, 

$$
x^\star = \eta^S_{\lambda/\|X_i\|_2^2}(g^\star) = \eta^S_{\lambda/\|X_i\|_2^2} \left(\frac{X_i^T (y-X_{-i} \beta_{-i}) }{X_i^T X_i}\right), 
$$
where $\eta_\epsilon$ is the *soft-thresholding* function given by

$$
\eta^S_\epsilon(x) = \begin{cases}
x - \epsilon & \textrm{ if } x > \epsilon \\
x + \epsilon & \textrm{ if } x < -\epsilon \\
0 & \textrm{ if } -\epsilon \leq x \leq \epsilon
\end{cases}.
$$

The soft-thresholding shrinks the value of $x$ until it hits zero (and then leaves it at zero). This is in contrast to the hard-thresholding function that sets large elements to $0$.

**Hard-thresholding:** 

$$
\eta^H_\epsilon(x) = x {\bf 1}_{|x| > \epsilon}.
$$

```{figure} images/fig_thres_hard.png
---
width: 300 px
---
```

**Soft-thresholding:** 

$$
\eta^S_\epsilon(x) = \textrm{sgn}(x) (|x| - \epsilon)_+
$$

```{figure} images/fig_thres_soft.png
---
width: 300 px
---
```

In conclusion, to solve the lasso problem using coordinate descent: 

1. Pick an initial point $\beta$.
2. Cycle through the coordinates and perform the updates

$$
\beta_i \rightarrow \eta^S_{\lambda/\|X_i\|_2^2} \left(\frac{2 X_i^T (y-X_{-i} \beta_{-i}) }{X_i^T X_i}\right).
$$

3. Continue until convergence (i.e., stop when the coordinates vary less than some threshold).
