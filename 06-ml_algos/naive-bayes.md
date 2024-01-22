## Naive Bayes

**Naive Bayes** is a classification algorithm based on Bayes' theorem, which is a statistical technique that uses probability to make predictions. This algorithm is very simple but effective and is widely used in various areas of Machine Learning.

The name *Naive* comes from the assumption that all features (predictor variables) in the data set are independent of each other (there is no correlation between them) given the value of the target variable. In other words, this assumption assumes that each characteristic contributes independently to the probability of belonging to a particular class.

### Bayes' Theorem

**Bayes' theorem** is a fundamental concept in probability that allows us to update our beliefs or probabilities about an event given new evidence. The formula on which this theorem is based is as follows:

$P(A|B) = {P(B|A) * P(A)} / {P(B)}$,

and where:
- $P(A|B)$ is the probability that event $A$ occurs given that we already know that event $B$ has occurred.
- $P(B|A)$ is the probability that event $B$ occurs given that we already know that event $A$ has occurred.
- $P(A)$ is the initial probability that event $A$ occurs before considering evidence $B$.
- $P(B)$ is the probability that $B$ occurs.

Moreover, $A$ is the variable to predict and $B$ is the predictor. Bayes' theorem allows us to adjust our original beliefs ($P(A)$) about an event, using new information ($P(B|A)$ and $P(B)$). Essentially, it helps us calculate the updated probability of an event occurring taking into account new evidence. It is a very useful tool in many fields, from science and medicine to Machine Learning and artificial intelligence, to make decisions and predictions based on observed data.

### Implementations

In `scikit-learn` there are three implementations of this model: `GaussianNB`, `MultinomialNB` and `BernoulliNB`. These implementations differ mainly in the type of data they can handle and the assumptions they make about the distribution of the data:

| | GaussianNB | MultinomialNB | BernoulliNB |
|-|------------|---------------|-------------|
| Data type | Continuous data. | Discrete data. | Binary data. |
| Distributions | Assumes that the data follow a normal distribution. | Assumes that the data follow a multinomial distribution. | Assumes that the data follow a Bernoulli distribution. |
| Common usage | Classification with numeric characteristics. | Classification with features representing discrete counts or frequencies. | Classification with binary features. |

If you have both numerical and categorical features in your data, there are different strategies, but the best one to preserve the usefulness and suitability of this model is to transform the categorical ones into numerical ones using coding techniques as we have seen above: `pd.factorize` of `Pandas`.

### Model hyperparameterization

We can easily build a decision tree in Python using the `scikit-learn` library and the `GaussianNB`, `MultinomialNB` and `BernoulliNB` functions. Some of its most important hyperparameters and the first ones we should focus on are:

- `alpha`: It is used to avoid zero probabilities on features that do not appear in the training set. A larger value adds more smoothing (only for `MultinomialNB` and `BernoulliNB`).
- `fit_prior`: Indicates whether to learn the a priori probabilities of the classes from the data or to use uniform probabilities (only for `MultinomialNB`).
- `binarize`: Threshold to normalize features. If a value is provided, the features are binarized according to that threshold; otherwise, it is assumed that the features are already binarized. If they were not and this hyperparameter is not used, the model may not perform well (only for `BernoulliNB`).

As you can see the hyperparameterization in this type of models is very small, so one way to optimize this type of models is, for example, to eliminate variables that are highly correlated (if variable $A$ and $B$ have a high correlation, one of them is eliminated), since in this type of models they have a double importance.
