# Random Forest

**What is Random Forest?**


**How does Random Forest differ from traditional Decision Tree algorithms?**

Random Forest is an ensemble method that uses bagged decision trees with random feature subsets chosen at each split point. It then either averages the prediction results of each tree (regression) or using votes from each tree (classification) to make the final prediction.

**What hyperparameters can be tuned for a random forest that are in addition to each individual tree's hyperparameters?**

Random forest is essentially bagged decision trees with random feature subsets chosen at each split point, so we have 2 new hyperparameters that we can tune:

num estimators - the number of decision trees in the forest

max features - maximum number of features that are evaluated for splitting at each node

**Are random forest models prone to overfitting? Why?**

No, random forest models are generally not prone to overfitting because the bagging and randomized feature selection tends to average out any noise in the model. Adding more trees does not cause overfitting since the randomization process continues to average out noise (more trees generally reduces overfitting in random forest).

In general, bagging algorithms are robust to overfitting.

Having said that, it is possible to overfit with random forest models if the underlying decision trees have extremely high variance. Extremely high depth and low min sample split, and a large percentage of features are considered at each split point. For example if every tree is identical, then random forest may overfit the data.


Source: 

https://www.dataquest.io/blog/top-10-machine-learning-algorithms-for-beginners/#:~:text=The%20first%205%20algorithms%20that,are%20examples%20of%20supervised%20learning.
