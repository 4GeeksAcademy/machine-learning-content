
## Evaluation metrics

### Evaluation metrics useful for clasification problems

1. Accuracy - measures the percentage of the time we correctly classify samples: (true positive + true negative)/ all samples

2. Precision - measures the percentage of the predicted members that were correctly classified: true positives/ (true positives + false positives)

3. Recall - measures the percentage of true members that were correctly classified by the algorithm: true positives/ (true positive + false negative)

4. F1 - measurement that balances accuracy and precision (or you can think of it as balancing Type I and Type II error)

5. AUC - describes the probability that a classifier will rank a randomly chosen positive instance higher than a randomly chosen negative one.

6. Gini - a scale and centered version of AUC

7. Log-loss - similar to accuracy but increases the penalty for incorrect classifications that are further away from their true class. For log-loss, lower values are better.

To see how to implement each of them see the scikit-learn documentation examples:

https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics

### Evaluation metrics useful for regression problems

1. Means squared error (MSE)- the average of the squared error of each prediction.

2. Root mean squared error (RMSE) - Square root of MSE.

3. Mean absolute error (MAE) - the average of the absolute error of each prediction.

4. Coefficient of determination (R^2) - proportion of variance in the target that is predictable from the features.

To see how to implement each of them see the scikit-learn documentation examples:

https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics

## Modeling errors

**What are the three types of error in a Machine Learning model?**

1. Bias - error caused by choosing an algorithm that cannot accurately model the signal in the data. The model is too general or was incorrectly selected. For example, selecting a simple linear regression to model highly non-linear data would result in error due to bias.

2. Variance - error from an estimator being too specific and learning relationships that are specific to the training set but do not generalize to new samples well. Variance can come from fitting too closely to noise in the data, and models with high variance are extremely sensitive to changing inputs.  For example, creating a decision tree that splits the training set until every leaf node only contains 1 sample.

3. Irreducible error - error caused by noise in the data that cannot be removed through modeling. For example, inaccuracy in data collection causes irreducible error.

**What is the bias-variance trade-off?**

**Bias** refers to an error from an estimator that is too general and does not learn relationships from a data set that would allow it to make better predictions.

**Variance** refers to an error from an estimator being too specific and learning relationships that are specific to the training set but will not generalize to new records well.

In short, the bias-variance trade-off is a trade-off between underfitting and overfitting. As you decrease bias, you tend to increase variance.

Our goal is to create models that minimize the overall error by careful model selection and tuning to ensure there is a balance between bias and variance, general enough to make good predictions on new data but specific enough to pick up as much signal as possible.

Source:

https://towardsdatascience.com/interpreting-roc-curve-and-roc-auc-for-classification-evaluation-28ec3983f077