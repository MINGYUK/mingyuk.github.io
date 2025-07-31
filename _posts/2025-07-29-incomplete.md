---
layout: post
title: Stable balancing weights - literature summary
date: 2025-07-29 00:01:00
description: Stable balancing weights is a weighting algorithm used in causal inference. Let's review the original literature.
tags:
  - weighint algorithm
  - machine learning
categories: machine-learning, causal-inference
pseudocode: false
toc:
  sidebar: true
featured: false
tabs: false
---

# Summary
Weighting algorithms often suffer weight instability, i.e. high variance. Stable balancing weights (SBW) is a weighting method with minimal variance. It looks for weights that balances the distribution within a pre-specified threshold.

# Introduction
Weighting algos have two goals:
 - balance the empirical distribution of observed covariates (remove bias due to observed confounders)
 - yield a stable estimate of the parameter of interest (minimize variance)

IPTW doesn't do both. Also, even with different methods that makes propensity score model less sensitive to model specification, the weights acquired is not intended to balance the distribution in the first place.

SBW directly aims to optimize the weights so that it balances the covariate distribution, including marginal distributions.

# The Estimation Problem
 - $\mathcal{P}$: target population with $N$ elements
 - $\mathcal{S}$: random sample from $\mathcal{P}$ of size $n$
 - $\mathcal{R}$: sample of size $\mathcal{r}$ respondents from $\mathcal{S}$
 - $Z_i$: response (treatment) indicator for unit $i$ in $\mathcal{S}$ ($Z_i = 1$ if unit $i$ responds)
 - $Y_i$: outcome variable
 - $X_{ip}$: $p$-th covariate of unit $X_i$

The parameter of interest we want to estimate is the population mean:
$$\overline{Y}_N = \frac{\sum_{i=1}^{N}{Y_i}}{N}$$

Assume the response (treatment) is missing completely at random (MCAR) in $\mathcal{S}$, then $$\hat{Y}_r = \frac{\sum_{i=1}^{r}{Y_i}}{r}$$ is an unbiased, consistent estimator of $\overline{Y}_N$.

This is a strong assumption, and a more realistic assumption is non-respondents missing at random (MAR). Under this assumption, $Z_i$ and $Y_i$ are both related to observed covariates $X_i$ but not any unobserved covaraite $U_i$. (refer to "statistical analysis with missing data" by Little and Rubin)

In this case, $$\hat{Y}_{\mathit{w}} = \frac{\sum_{i=1}^{r}{\mathit{w}_iY_i}}{r}$$ is the unbiased, consistent estimator of $\overline{Y}_N$, given $\mathbf{w}$ appropriately adjusts for $X$. The most common way to calculate the weights is to fit a model that estimates the probability of responding (getting treatment) and inverting them. This is called the propensity score. An appealing aspect of using the propensity score is that it tends to balance the distribution of the observed covariates.

However, this is a stochastic property (relies on the law of large numbers), and is not guaranteed even when the true mechanism of missing response is known. In reality the true mechanism is almost always unknown, which makes the balancing harder. Also, it is more desirable to balance features other than the covariate means, such as the marginal distribution. Finally, the weights can be unstable. The variance of $\hat{Y}_w$ is quadratic to $cv(\mathit{w}_i)$, the coefficient of variance. Even if a few units have weights close to zero, this will result in high variance of the estimate.

# The Target Convex Optimization Problem
Rather than finding the weights for predicting response/treatment, SBW directly finds the weights with minimal variance that balances the covariates. This is the optimization problem that describes SBW:

$$
\displaylines{
\min_{w}{\|w - \bar{w}\|_2^2}\\
\text{subject to } |\mathbf{w}^T\mathbf{X}_{\mathcal{R}_{p}} - \bar{X}_{\mathcal{S}_{p}}| \le \delta_p, p=1, ..., P\\
\mathbf{1}^T\mathbf{w} = 1,\\
\mathbf{w} \ge 0
}
$$

$\mathbf{w}$ is a vector of size $r$, and $\bar{w}$ is the mean value vector of the weights. $\delta_{p}$ is a hyperparameter, defined by the user. This optimization effectively minimizes the coefficient of variance of the weights while balancing the covariates. Also, by manipulating $X_{\mathcal{R}}$, the constraint can balance other statistics such as variance. To do this, we augment $X_{\mathcal{R}}$ with additional covariate matrix $$\tilde{X}_{\mathcal{R}} = X_{\mathcal{R}}^2$$.
