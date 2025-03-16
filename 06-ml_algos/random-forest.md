---
description: >-
  Discover the power of Random Forests in machine learning! Learn how this
  robust method enhances classification and regression tasks. Click to explore!
---
## Random Forests

A **random forest** is a Machine Learning method used for classification and regression tasks. It is a type of learning in which *model ensembling* is used to combine the predictions of multiple decision trees to generate a more accurate and robust output.

Each tree in a random forest is constructed independently using a random subset of the training data. Then, to make the prediction, each tree in the forest makes its own prediction, and the final prediction is taken by majority voting in the case of classification, or averaging in the case of regression.

This approach helps to overcome the problem of overfitting, which is common with individual decision trees.

### Structure

A random forest is a collection of decision trees. Each of these trees is a model consisting of decision nodes and leaves. The decision nodes, recall, are points where decisions are made based on certain attributes or characteristics, and the leaves are the final outcomes or predictions.

Thus, to build each decision tree, the random forest selects a random subset of the training data. This process is called **bagging** or **bootstrap aggregating**.

In addition to selecting random subsets of data, the random forest also selects a random subset of the features from each tree. This adds another layer of randomness to the model, which helps to increase the diversity among the trees and improve the overall model robustness.

![A random forest](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/random_forest.PNG?raw=true)

Once trained, each decision tree within the random forest performs its own prediction. For classification problems, the class that obtains the most votes among all the trees is selected as the final prediction. For regression problems, the final prediction is obtained by averaging the predictions of all the trees.

The random forest structure, with its combination of randomness and aggregation, helps to create a robust model that is less prone to overfitting the training data compared to a single decision tree.

### Model hyperparameterization

We can easily build a decision tree in Python using the `scikit-learn` library and the `RandomForestClassifier` and `RandomForestRegressor` functions. Some of its most important hyperparameters, and the first ones we should focus on are:

- `n_estimators`: This is probably the most important hyperparameter. It defines the number of decision trees in the forest. In general, a larger number of trees increases the accuracy and makes the predictions more stable, but it can also slow down the computation time considerably.
- `bootstrap`: This hyperparameter is used to control whether bootstrap samples (sampling with replacement) are used for tree construction.
- `max_depth`: The maximum depth of the trees. This is essentially how many splits the tree can make before making a prediction.
- `min_samples_split`: The minimum number of samples needed to split a node in each tree. If set to a high value, it prevents the model from learning too specific relationships and thus helps prevent overfitting.
- `min_samples_leaf`: The minimum number of samples to have in a leaf node in each tree.
- `max_features`: The maximum number of features to consider when looking for the best split within each tree. For example, if we have 10 features, we can choose to have each tree consider only a subset of them when deciding where to split.

As we can see, only the first two hyperparameters refer to the random forest, while the rest are related to the decision trees. Another very important hyperparameter is the `random_state`, which controls the random generation seed. This attribute is crucial to ensure replicability.
