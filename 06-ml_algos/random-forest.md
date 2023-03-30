# Random Forest

Ensembling is another type of supervised learning. It combines the predictions of multiple machine learning models that are individually weak to produce a more accurate prediction on a new sample. By combining individual models, the ensemble model tends to be more flexible🤸‍♀️ (less bias) and less data-sensitive🧘‍♀️ (less variance).

The idea is that ensembles of learners perform better than single learners.

In the next two lessons we will learn about two ensemble techniques, bagging with random forests and boosting with XGBoost.

**What does bagging mean?**

Training a bunch of individual models in a parallel way. Each model is trained by a random subset of the data.

**How does the Random Forest model work?**

To understand the random forest model, we first learned about the decision tree, the basic building block of a random forest. We all use decision trees in our daily life, and even if you don’t know it, you’ll recognize the process.

![decision_tree_daily_life](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/decision_tree_daily_life.jpg?raw=true)

Random forest, like its name implies, consists of a large number of individual decision trees that operate as an ensemble. Each individual tree in the random forest spits out a class prediction and the class with the most votes becomes our model’s prediction.

Some deeper explanation:

Unlike a decision tree, where each node is split on the best feature that minimizes error, in Random Forests, we choose a random selection of features for constructing the best split. The reason for randomness is: even with bagging, when decision trees choose the best feature to split on, they end up with similar structure and correlated predictions. But bagging after splitting on a random subset of features means less correlation among predictions from subtrees.

The number of features to be searched at each split point is specified as a parameter to the Random Forest algorithm.

Thus, in bagging with Random Forest, each tree is constructed using a random sample of records and each split is constructed using a random sample of predictors.

To clarify the difference between them, random Forest is an ensemble method that uses bagged decision trees with random feature subsets chosen at each split point. It then either averages the prediction results of each tree (regression) or using votes from each tree (classification) to make the final prediction.

> The reason why they work so well: 'A large number of relatively uncorrelated models (trees) operating as a committee will outperform any of the individual constituent models'. Low correlation is the key. 

## What hyperparameters can be tuned for a random forest that are in addition to each individual tree's hyperparameters?

Always a good place to start is reading the documentation in scikit learn: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

The most important settings are: 

- num estimators - the number of decision trees in the forest

- max features - maximum number of features that are evaluated for splitting at each node

But we can try adjusting a wide range of values in other hyperparameters like:

- max_depth = max number of levels in each decision tree

- min_samples_split = min number of data points placed in a node before the node is split

- min_samples_leaf = min number of data points allowed in a leaf node

- bootstrap = method for sampling data points (with or without replacement)

Let's see how could we implement a RandomizedSearchCV to find optimal hyperparameters:

```py

from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)
{'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
 
```

On each iteration, the algorithm will choose different combinations of the features. Altogether, there are 2 * 12 * 2 * 3 * 3 * 10 = 4320 settings! However, the benefit of a random search is that we are not trying every combination, but we are selecting at random to sample a wide range of values.

## Are random forest models prone to overfitting? Why?

No, random forest models are generally not prone to overfitting because the bagging and randomized feature selection tends to average out any noise in the model. Adding more trees does not cause overfitting since the randomization process continues to average out noise (more trees generally reduces overfitting in random forest).

In general, bagging algorithms are robust to overfitting.

Having said that, it is possible to overfit with random forest models if the underlying decision trees have extremely high variance. Extremely high depth and low min sample split, and a large percentage of features are considered at each split point. For example if every tree is identical, then random forest may overfit the data.

**How can my random forest make accurate class predictions?**

- We need features that have at least some predictive power.

- The trees of the forest and their predictions need to be uncorrelated (at least low correlations). Features and hyperparameters selected will impact ultimate correlations.   



Source: 

https://www.dataquest.io/blog/top-10-machine-learning-algorithms-for-beginners/#:~:text=The%20first%205%20algorithms%20that,are%20examples%20of%20supervised%20learning.

https://towardsdatascience.com/understanding-random-forest-58381e0602d2

https://towardsdatascience.com/basic-ensemble-learning-random-forest-adaboost-gradient-boosting-step-by-step-explained-95d49d1e2725

https://williamkoehrsen.medium.com/random-forest-simple-explanation-377895a60d2d

https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
