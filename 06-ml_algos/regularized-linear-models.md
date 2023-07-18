## Regularized Linear Models

A **regularized linear model** is a version of a linear model that includes an element in its function to avoid overfitting and improve the learning capability of the model.

Generally speaking, a linear model (like the one we saw in the previous module) tries to find the relationship between the input variables and the output variable. However, if a linear model has too many parameters or if the data are very noisy, it can happen that the model fits the training data too well, producing a clear overfit and making it difficult to generalize well to new data.

To avoid this problem, regularized linear models add an extra term to penalize coefficient values that are too large. These models are linear regressions like those seen in the previous module but with the addition of a regularization term. The two types of models are:

- **Lasso regularized linear model** (*L1*): Adds a penalty equal to the absolute value of the magnitude of the coefficients. May result in coefficients equal to zero, indicating that the corresponding feature is not used in the model.
- **Ridge regularized linear model** (*L2*): Adds a penalty equal to the square of the magnitude of the coefficients. This tends to reduce the coefficients but does not make them exactly zero, so all features remain in the model.

Both techniques attempt to limit or "penalize" the size of the coefficients in the model. Imagine that we are fitting a line to points on a graph:

- **Linear regression**: We only care about finding the line that best fits the points.
- **Ridge linear regression**: We try to find the line that fits best, but we also want to keep the slope of the line as small as possible.
- **Lasso linear regression**: As with Ridge, we try to fit the line and keep the slope small, but Lasso can take the slope to zero if that helps fit the data. This is like "cherry-picking" which variables are important and which are not, because he can reduce the importance of some variables to zero.

### Model parameterization

We can easily build a regularized linear model in Python using the `scikit-learn` library and the `Lasso` and `Ridge` functions. Some of its most important parameters and the first ones we should focus on are:

- `alpha`: This is the regularization hyperparameter. It controls how much we want to penalize high coefficients. A higher value increases the regularization and therefore the model coefficients tend to be smaller. Conversely, a lower value reduces it and allows higher coefficients. The default value is 1.0 and its range of values goes from 0.0 to infinity.
- `max_iter`: This is the maximum number of iterations of the model. 

Another very important parameter is the `random_state`, which controls the random generation seed. This parameter is crucial to ensure replicability.

### Model usage in Python

You can easily use `scikit-learn` to program these methods after the EDA:

#### Lasso

```py
from sklearn.linear_model import Lasso

# Load of train and test data
# These data must have been standardized and correctly processed in a complete EDA


lasso_model = Lasso(alpha = 0.1, max_iter = 300)

lasso_model.fit(X_train, y_train)

y_pred = lasso_model.predict(y_test)
```

#### Ridge

```py
from sklearn.linear_model import Ridge

# Load of train and test data
# These data must have been standardized and correctly processed in a complete EDA

ridge_model = Ridge(alpha = 0.1, max_iter = 300)

ridge_model.fit(X_train, y_train)

y_pred = ridge_model.predict(y_test)
```