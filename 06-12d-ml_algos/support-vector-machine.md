# Support Vector Machine

What is SVM...

SVM can not produce a probability score along with its classification prediction.

**What parameters can we tune for Support Vector Machine?**

The hyperparameters that you can commonly tune for Support Vector Machine are:

-Regularization/cost parameter

-Kernel

-Degree of polynomial (if you are using a polynomial kernel)

-Gamma (modifies the influence of nearby points on the support vector for Gaussian RBF kernels)

-Coef0 (influences impact of high vs low  degree polynomials for polynomial or sigmoid kernels)

-Epsilon (a margin term used for SVM regressions)

**What are some possible uses of SVM models?**

- Linear Classification

- Non linear Classification

- Linear Regression

- Non linear Regression

**What common kernels can you use for SVM?**

1. Linear

2. Polynomial

3. Gaussian RBF

4. Sigmoid

**Why is it important to scale feature before using SVM?**

SVM tries to fit the widest gap between all classes, so unscaled features can cause some features to have a significantly larger or smaller impact on how the SVM split is created.



