# Regularized Linear Models


Regularized linear models are...

1. Ridge Regression - linear regression that adds L2-norm penalty/regularization term to the cost function.

2. Lasso - linear regression that adds L1-norm penalty/regularization term to the cost function.

3. Elastic Net - linear regression that adds mix of both L1- and L2-norm penalties terms to the cost function.

**What hyperparameters can be tuned in regularized linear models?**

You can tune the weight of the regularization term for regularized models (typically denoted as alpha), which affect how much the models will compress features.

-alpha = 0 ---> regularized model is identical to original model.

-alpha = 1 ---> regularized model reduced the original model to a constant value.

**Regularized models performance**

Regularized models tend to outperform non-regularized linear models, so it is suggested that you at least try using ridge regression.

Lasso can be effective when you want to automatically do feature selection in order to create a simpler model but can be dangerous since it may be erratic  and remove features that contain useful signal.

Elastic net is a balance of ridge and lasso, and it can be used to the same effect as lasso with less erratic behaviour.