# Feature Selection



***When should we reduce the number of features used by our model?**

Some instances when features selection is necessary:

- When there is strong collinearity between features

- There are an overwhelming number of features

- There is not enough computational power to process all features.

- The algorithm forces the model to use all features, even when they are not useful (most often in parametric or linear models)

- When we wish to make the model simpler for any reason. For example to make it easier to explain, less computational power needed.

**When is feature selection unnecesary?**

Some instances when feature selection is not necessary:

- There are relatively few features

- All features contain useful and important signal

- There is no collinearity between features

- The model will automatically select the most useful features

- The computing resources can handle processing all of the features

- Thoroughly explaining the model to a non-technical audience is not critical

**What are the three types of feature selection methods?**

- Filter Methods - feature selection is done independent of the learning algorithm, before any modelling is done. One example is finding the correlation between every feature and the target and throwing out those that don't meet a threshold. Easy, fast, but naive and not as performant as other methods.

- Wrapper Methods - train models on subsets of the features and use the subset that results in the best performance. Examples are Stepwise or Recursive feature selection. Advantages are that it considers each feature in the context of the other features, but can be computationally expensive.

- Embedded Methods - learning algorithms have built-in feature selection. For example: L1 regularization.
