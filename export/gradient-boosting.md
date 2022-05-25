# Gradient Boosting

**How does gradient boosting machines differ from traditional decision tree algorithms?**

Gradient boosting involves using multiple weak predictors (decision trees) to create a strong predictor. Specifically, it includes a loss function that calculates the gradient of the error with regard to each feature and then iteratively creates new decision trees that minimize the current error. More and more trees are added to the current model to continue correcting error until improvements fall below some minimum threshold or a pre-decided number of trees have been created.

**What hyperparameters can be tuned in gradient boosting that are in addition to each individual tree's hyperparameters?**

The main hyperparameters that can be tuned with GBM models are:

- Loss function - loss function to calculate gradient of error

- Learning rate - the rate at which new trees correct/modify the existing predictor

- Num estimators - the total number of trees to produce for the final predictor

Additional hyperparameters specific to the loss function 

Some specific implementations, for example stochastic gradient boosting, may have additional hyperparameters such as subsample size (subsample size affects the randomization in stochastic variations).

**How can we reduce overfitting when doing gradient boosting?**

Reducing the learning rate or reducing the maximum number of estimators are the two easiest ways to deal with gradient boosting models that overfit the data.

With stochastic gradient boosting, reducing subsample size is an additional way to combat overfitting.

Boosting algorithms tend to be vulnerable to overfitting, so knowing how to reduce overfitting is important.
