
# Hypertuning Machine Learning Algorithms

### What are two common ways to automate hyperparameter tuning?

1. Grid Search - test every possible combination of pre-defined hyperparameter values and select the best one.

2. Randomized Search - randomly test possible combinations of pre-defined hyperparameter values and select the best tested one.

**What are the pros and cons of grid search?**

**Pros:**

Grid Search is great when we need to fine-tune hyperparameters over a small search space automatically. For example, if we have 100 different datasets that we expect to be similar, like solving the same problem repeatedly with different populations. We can use grid search to automatically fine-tune the hyperparameters for each model.

**Cons:** 

Grid Search is computationally expensive and inefficient, often searching over parameter space that has very little chance of being useful, resulting it being extremely slow. It's especially slow if we need to search a large space since it's complexity increases exponentially as more hyperparameters are optimized.

**What are the pros and cons of randomized search?**

**Pros:**

Randomized search does a good job finding near-optimal hyperparameters over a very large search space relatively quickly and doesn't suffer from the same exponential scaling problem as grid search.

**Cons:**

Randomized search does not fine-tune the results as much as grid search does since it tipically does not test every possible combination of parameters.
