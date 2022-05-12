
### Decision Trees

**What are decision trees?**




**What are the common uses of decision tree algorithms?**

1. Classification 

2. Regression

3. Measuring feature importance 

4. Feature selection


**What are the main hyperparameters that you can tune for decision trees?**

Generally speaking, decision trees have the following parameters:

- max depth - maximum tree depth

- min samples split - minimum number of samples for a node to be split

- min samples leaf - minimum number of samples for each leaf node

- max leaf nodes - the maximum number of leaf nodes in the tree

- max features - maximum number of features that are evaluated for splitting at each node (only valid for algorithms that randomize features considered at each split)

The traditional decision tree is greedy and looks at all features at each split point, but many modern implementations allow splitting on randomized features (as seen in scikit learn) so max features may or may not be a tuneable hyperparameter.


**How each hyperparameter affects the model's ability to learn?**

- max depth: increasing max depth will decrease bias and increase variance 

- min samples split: increasing min samples split increases bias and decreases variance

- min samples leaf: increasing min samples leaf increases bias and decreases variance

- max leaf nodes: decreasing max leaf node increases bias and decreases variance

- max features: decreasing maximum features increases bias and decreases variance

There may be instances when changing hyperparameters has no effect on the model.

**What metrics are usually used to compute splits?**

Gini impurity or entropy. Both generally produce similar results.


**What is Gini impurity?**

Gini impurity is a measurement of how often a randomly chosen record would be incorrectly classified if it was randomly classified using the distribution of the set of samples.

A low Gini (near 0) means most records from the sample are in the same class.

A high Gini (maximum of 1 or less, depending on number of classes) = records from sample are spread evenly across classes.

**What is entropy?**

Entropy is the measure of the purity of members among non-empty classes. It is very similar to Gini in concept, but a slightly different calculation.

Low entropy (near 0) means most records from the sample are in the same class.

High entropy (maximum of 1) means records from sample are spread evenly across classes.

**Are decision trees parametric or non-parametric models?**

Non-parametric. The number of model parameters is not determined before creating the model.

**What are some ways to reduce overfitting with decision trees?**

- Reduce maximum depth

- Increase min samples split

- Balance data to prevent bias toward dominant classes

- Increase the number of samples

- Decrease the number of features

**How is feature importance evaluated in decision-tree based models?**

The features that are split on most frequently and are closest to the top of the tree, thus affecting the largest number of samples, are considered to be the most important.
