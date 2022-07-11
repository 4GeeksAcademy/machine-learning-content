# Support Vector Machine

## What hyperparameters can we tune for SVM?

The hyperparameters that we commonly tune for SVM are:

-Regularization /cost parameter

-Kernel

-Degree of polynomial (if using a polynomial kernel)

-Gamma (modifies the influence of nearby points on the support vector for Gaussian RBF kernels)

-Coef() (influences impact of high vs low degree polynomials for polynomial or sigmoid kernels)

-Epsilon (a margin term used for SVM regressions)

**What are some possible uses of SVM models?**

Support Vector Machine can be used for:

-Linear classification

-Non linear classification

-Linear Regression

-Non linear Regression

**What common kernels can be used for SVM?**

1. Linear 

2. Polynomial

3. Gaussian RBF

4. Sigmoid

**Why is it important to scale features before using SVM?**

SVM tries to fit the widest gap between all classes, so unscaled features can cause some features to have a significantly larger or smaller impact on how the SVM split is created.

**Can SVM produce a probability score along with its classification prediction?**

No.


