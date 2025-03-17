---
description: >-
  Master feature selection techniques to enhance model performance and
  interpretability. Discover when and how to reduce features effectively!
---
# Feature Selection

**What is feature selection?**

The objective of feature selection is to enhance the interpretability of models, speed up the learning process and increase the predictive performance.

**When should we reduce the number of features used by our model?**

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

    - Basic method.

    - Correlation method.
    
    - Statistical methods ( Information gain / Chi Square / ANOVA ).  

- Wrapper Methods - train models on subsets of the features and use the subset that results in the best performance. Examples are Stepwise or Recursive feature selection. Advantages are that it considers each feature in the context of the other features, but can be computationally expensive.

    - Forward selection.

    - Backward elimination.

    - Exhaustive search.


- Embedded Methods - learning algorithms have built-in feature selection. For example: L1 regularization.

    - LASSO Regularization.
    
    - Feature importances.


Use the following [notebook](https://github.com/priyamnagar/feature_selection_titanic/blob/master/Titanic.ipynb) to see how to apply each of this methods in a dataset that has already been split into training and validation sets.

Consider the folllowing links for statistical methods if you plan to apply feature selection:

-[Chi square test](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html#sklearn.feature_selection.chi2)

-[Anova](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html#sklearn.feature_selection.f_regression)

-[Mutual Information](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html#sklearn.feature_selection.mutual_info_classif)


