# Regularized Linear Models

Before explaining regularized linear models, let's recap some important information about linear regression.

Every machine learning problem is basically an optimization problem. That is, you wish to find either a maximum or a minimum of a specific function. The function that you want to optimize is usually called the loss function (or cost function). The loss function is defined for each machine learning algorithm you use, and this is the main metric for evaluating the accuracy of your trained model.

This is the most basic form of a loss for a specific data-point, that is used mostly for linear regression algorithms:

$l = ( Ŷi- Yi)^2$

Where :

- Ŷi is the predicted value

- Yi is the actual value

The loss function as a whole can be denoted as:

$L = ∑( Ŷi- Yi)^2$

This loss function, in particular, is called quadratic loss or least squares. We wish to minimize the loss function (L) as much as possible so the prediction will be as close as possible to the ground truth.

> Remember, every machine learning algorithm defines its own loss function according to its goal in life

## Overcoming overfit with regularization

We finished the last lesson talking about the importance of avoiding overfitting. One of the most common mechanisms for avoiding overfit is called regularization. Regularized machine learning model, is a model that its loss function contains another element that should be minimized as well. Let’s see an example:

$L = ∑( Ŷi- Yi)^2 + λ∑ β2$



1. **Ridge Regression** - linear regression that adds L2-norm penalty/regularization term to the cost function. The λ parameter is a scalar that should be learned as well, using cross validation. A super important fact we need to notice about ridge regression is that it enforces the β coefficients to be lower, but it does not enforce them to be zero. That is, it will not get rid of irrelevant features but rather minimize their impact on the trained model.

2. **Lasso** - linear regression that adds L1-norm penalty/regularization term to the cost function. The only difference from Ridge regression is that the regularization term is in absolute value. But this difference has a huge impact on the trade-off we’ve discussed before. Lasso method overcomes the disadvantage of Ridge regression by not only punishing high values of the coefficients β but actually setting them to zero if they are not relevant. Therefore, you might end up with fewer features included in the model than you started with, which is a huge advantage.

3. Elastic Net - linear regression that adds mix of both L1- and L2-norm penalties terms to the cost function.

## What hyperparameters can be tuned in regularized linear models?

You can tune the weight of the regularization term for regularized models (typically denoted as alpha), which affect how much the models will compress features.

-alpha = 0 ---> regularized model is identical to original model.

-alpha = 1 ---> regularized model reduced the original model to a constant value.

**Regularized models performance**

Regularized models tend to outperform non-regularized linear models, so it is suggested that you at least try using ridge regression.

Lasso can be effective when you want to automatically do feature selection in order to create a simpler model but can be dangerous since it may be erratic  and remove features that contain useful signal.

Elastic net is a balance of ridge and lasso, and it can be used to the same effect as lasso with less erratic behaviour.


Source:

https://medium.com/hackernoon/practical-machine-learning-ridge-regression-vs-lasso-a00326371ece