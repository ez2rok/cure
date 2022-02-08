# CURE
Implement the clustering algorithm Clustering via Uncoupled  REgression (CURE) from Wang's paper [Efficient Clustering for Stretched Mixtures: Landscape and Optimality](https://arxiv.org/abs/2003.09960).

<p align="center">
    <img src="/reports/figures/experiment1/cure_animation.gif" alt="GIF of CURE" width="400" height="400" />
</p>


# Problem Description

## Motivation
Many traditional clustering algorithms struggle to cluster elliptically distributed data. KNN, for example, assumes the data is spherically distributed and performs poorly when data is elliptically distributed. CURE seeks to solve this problem by creating a clustering algorithm that excels at clustering elliptically distributed data.

## Equations

CURE seeks to find the weights that minimize the loss function

<p align="center">
<img 
     src="https://latex.codecogs.com/svg.image?\boldsymbol{\beta}^*&space;=&space;&space;&space;&space;&space;\underset{&space;\boldsymbol{\beta}&space;\in&space;\mathbb{R}^{d}}{\arg\min}&space;&space;&space;&space;\bigg\{&space;&space;&space;&space;&space;\frac{1}{n}&space;\sum_{i=1}^{n}&space;&space;&space;&space;f(&space;\boldsymbol{\beta}^T&space;\boldsymbol{X_i})&space;&space;&space;&space;&plus;&space;&space;&space;&space;\frac{1}{2}&space;(\boldsymbol{\beta}^T&space;\hat{\boldsymbol{\mu_0}})^2&space;&space;&space;&space;\bigg\}" title="\boldsymbol{\beta}^* = \underset{ \boldsymbol{\beta} \in \mathbb{R}^{d}}{\arg\min} \bigg\{ \frac{1}{n} \sum_{i=1}^{n} f( \boldsymbol{\beta}^T \boldsymbol{X_i}) + \frac{1}{2} (\boldsymbol{\beta}^T \hat{\boldsymbol{\mu_0}})^2 \bigg\}"
     alt="CURE Loss Function"
     />
</>

where 
    <img src="https://latex.codecogs.com/svg.image?\boldsymbol{\beta}&space;\in&space;\mathbb{R}^d" title="\boldsymbol{\beta} \in \mathbb{R}^d" /> 
are the weights, 
    <img src="https://latex.codecogs.com/svg.image?\boldsymbol{\beta}^{*}&space;\in&space;\mathbb{R}^d" title="\boldsymbol{\beta}^{*} \in \mathbb{R}^d" />
are the optimal weights that minimize the above equation, 
    <img src="https://latex.codecogs.com/svg.image?\boldsymbol{X_i}&space;\in&space;\mathbb{R}^d" title="\boldsymbol{X_i} \in \mathbb{R}^d" />
is a sample data point with 
    <img src="https://latex.codecogs.com/svg.image?d" title="d" />
features, 
    <img src="https://latex.codecogs.com/svg.image?\boldsymbol{X}&space;\in&space;\mathbb{R}^{n&space;\times&space;d}" title="\boldsymbol{X} \in \mathbb{R}^{n \times d}" />
is a matrix of 
    <img src="https://latex.codecogs.com/svg.image?n" title="n" />
datapoints each with
    <img src="https://latex.codecogs.com/svg.image?d" title="d" />
features, and 
    <img src="https://latex.codecogs.com/svg.image?\hat{\boldsymbol{\mu_0}}&space;=&space;\frac{1}{n}&space;\sum\nolimits_{i=1}^{n}&space;\boldsymbol{X_i}" title="\hat{\boldsymbol{\mu_0}} = \frac{1}{n} \sum_{i=1}^{n} \boldsymbol{X_i}" />
is the value of the average data point. To get 
    <img src="https://latex.codecogs.com/svg.image?\boldsymbol{X}" title="\boldsymbol{X}" />
we preappend a column of ones 
    <img src="https://latex.codecogs.com/svg.image?\boldsymbol{1}&space;\in&space;\mathbb{R}^n" title="\boldsymbol{1} \in \mathbb{R}^n" />
to the data (which is really 
    <img src="https://latex.codecogs.com/svg.image?\mathbb{R}^{n&space;\times&space;(d&space;-&space;1)}" title="\mathbb{R}^{n \times (d - 1)}" />
) to give us an intercept term. 

The discriminative function <img src="https://latex.codecogs.com/svg.image?f&space;:&space;\mathbb{R}&space;\rightarrow&space;\mathbb{R}" title="f : \mathbb{R} \rightarrow \mathbb{R}" /> is defined as

<img src="https://latex.codecogs.com/svg.image?f(x)=\begin{cases}&space;&space;&space;&space;&space;&space;&space;&space;h(x)&space;&space;&space;&space;&space;&space;&space;&space;&&space;&space;&space;&space;&space;&space;&space;&space;&space;|x|&space;\leq&space;a&space;&space;&space;&space;&space;&space;&space;&space;&space;\\&space;&space;&space;&space;&space;&space;&space;&space;f(a)&space;&plus;&space;h'(a)&space;(|x|&space;-&space;a)&space;&space;&space;&space;&space;&space;&space;&space;&space;&plus;&space;\frac{h''(a)}{2}&space;(|x|&space;-&space;a)^2&space;&space;&space;&space;&space;&space;&space;&space;&space;-&space;\frac{h''(a)}{6(b-a)}&space;(|x|&space;-&space;a)^3&space;&space;&space;&space;&space;&space;&space;&space;&&space;&space;&space;&space;&space;&space;&space;&space;&space;a&space;<&space;|x|&space;\leq&space;b&space;&space;&space;&space;&space;&space;&space;&space;&space;\\&space;&space;&space;&space;&space;&space;&space;&space;f(b)&space;&space;&space;&space;&space;&space;&space;&space;&space;&plus;&space;\Big[&space;h(a)&space;&plus;&space;\frac{b-a}{2}&space;h''(a)&space;\Big]&space;(|x|&space;-&space;b)&space;&space;&space;&space;&space;&space;&space;&space;&&space;&space;&space;&space;&space;&space;&space;&space;&space;|x|&space;>&space;b&space;&space;&space;&space;&space;&space;&space;&space;\\&space;&space;&space;&space;\end{cases}" title="f(x)=\begin{cases} h(x) & |x| \leq a \\ f(a) + h'(a) (|x| - a) + \frac{h''(a)}{2} (|x| - a)^2 - \frac{h''(a)}{6(b-a)} (|x| - a)^3 & a < |x| \leq b \\ f(b) + \Big[ h(a) + \frac{b-a}{2} h''(a) \Big] (|x| - b) & |x| > b \\ \end{cases}" />
    
    
$$
f(x)
=
\begin{cases}
        h(x)
        & 
        |x| \leq a 
        \\
        f(a) + h'(a) (|x| - a) 
        + \frac{h''(a)}{2} (|x| - a)^2 
        - \frac{h''(a)}{6(b-a)} (|x| - a)^3
        & 
        a < |x| \leq b 
        \\
        f(b) 
        + \Big[ h(a) + \frac{b-a}{2} h''(a) \Big] (|x| - b)
        & 
        |x| > b
        \\
    \end{cases}
$$
where $b > a > 1$ and $h(x) = \frac{(x^2-1)^2}{4}$

## Explanation

We embed the data $\boldsymbol{X}$ into 1D space via the dot product $x = \boldsymbol{\beta}^T \boldsymbol{X_i}$. Then we plug $x$ into the discriminative function $f$ to separate, ie. discriminate, the embedded data into two clusters. How does this work? $f$ and $h$ both have minimums at both x=±1 so minimizing these equations will map many of our datapoints to x=1 and many of them to x=-1, resulting in two different clusters. Because $h$ becomes huge for large $x$ values ($h$ is quartic), we construct $f$ which is just like $h$ except that when $x$ is too big we clip its growth with linear functions. More explitically, $f$ has three parts. When $x$ is small, $|x| \leq a$, we will minimize $h$ which has two valleys around ±1. When $x$ is too big, $|x| > b$, we will minimize a linear function so our values don't blow up. When $x$ is somewhere in between, $a < |x| \leq b$, we use a cubic spline to connect the valleys to the linear function.

![Discriminate functions](/reports/figures/discriminant.png)

So $\frac{1}{n} \sum_{i=1}^{n} f( \boldsymbol{\beta}^T \boldsymbol{X_i})$ embeds the data $\boldsymbol{X}$ in 1D space and computes on average how well the data is separated into two clusters located at x=±1. The lower this value, the better. However, we can minimize this function by clustering all of the datapoints into a single cluster. To avoid this trivial solution, we use the penalty term $\frac{1}{2} (\boldsymbol{\beta}^T \hat{\boldsymbol{\mu_0}})^2$. This term encourages the data $\boldsymbol{X}$ to be evenly split between the two clusters at x=±1 as this term has the lowest value when $\hat{\boldsymbol{\mu_0}} = 0$ which only occurs when the data is evenly split between both clusters. 

Putting it all together, the loss function embeds the data $\boldsymbol{X} $ in 1D space and computes on average how well the data is separated into two evenly sized clusters located at x=±1. We minimize this score and record the weights, $\boldsymbol{\beta}^*$, that result in this minimization.

## Clustering
Once we have computed $\boldsymbol{\beta}^*$, the clustering comes into play:
\begin{align}
    y^{\text{pred}}_i
    &= 
    \text{sgn}( \boldsymbol{\beta}^* \boldsymbol{X_i} ) 
\end{align}
where $y^{\text{pred}}_i$ is the predicted label of the $i$th sample. This function puts all positive datapoints into one cluster and all negative datapoints into a different cluster. And these are our clusters! That's it.

# Coding Overview

I designed CURE so it resembles many of the classifiers in `scikit-learn`. And I did this by creating a `CURE` class with `__init__` and three methods.

#### \_\_init__
When you initialize the class, you specify the the random seed and the parameters `a, b`.

#### fit()
When you call `fit`, it returns the weights that result in the best performance by minimizing the loss function. It relies upon `scikit-learn`'s minimization function.

#### predict()
When you call `predict`, it uses the weights computed in `fit` to predict a label for each datapoint. It uses the equation
$ y^{\text{pred}}_i = \text{sgn}( \boldsymbol{\beta}^* \boldsymbol{X_i} ) $.

#### fit_predict()
This calls both `fit` and `predict` on the training data.
