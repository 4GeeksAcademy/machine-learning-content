# Model Hyperparameter Optimization

## What is a model hyperparameter?

A model hyperparameter is the parameter whose value is set before the model starts training. They cannot be learned by fitting the model to the data.

Examples of model hyperparameters in different models:

- Learning rate in gradient descent

- Number of iterations in gradient descent

- Number of layers in a Neural Network

- Number of neurons per layer in a Neural Network

- Number of clusters(k) in k means clustering

## Difference between parameter and hyperparameter

A model parameter is a variable of the selected model which can be estimated by fitting the given data to the model. For example in linear regression, the slope and the intercept of the line are two parameters estimated by fitting a straight line to the data by minimizing the RMSE.

![parameter_vs_hyperparameter](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/parameter_vs_hyperparameter.jpg?raw=true)

And, as we already mentioned, a model hyperparameter value is set before the model start training and they cannot be learned by fitting the model to the data.

The best part is that you get a choice to select these for your model. Of course, you must select from a specific list of hyperparameters for a given model as it varies from model to model. 

Often, we are not aware of optimal values for hyperparameters which would generate the best model output. So, what we tell the model is to explore and select the optimal model architecture automatically. This selection procedure for hyperparameter is known as Hyperparameter Tuning.

## What are two common ways to automate hyperparameter tuning?

Hyperparameter tuning is an optimization technique and is an essential aspect of the machine learning process. A good choice of hyperparameters may make your model meet your desired metric. Yet, the plethora of hyperparameters, algorithms, and optimization objectives can lead to an unending cycle of continuous optimization effort.

1. Grid Search - test every possible combination of pre-defined hyperparameter values and select the best one.

Example:

```py

#import libraries
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV

#load the data
iris = datasets.load_iris()

#establish parameters
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}

#choose the model
svc = svm.SVC()

#Search all possible combinations
clf = GridSearchCV(svc, parameters)
clf.fit(iris.data, iris.target)

#get the hyperparameter keys
sorted(clf.cv_results_.keys())

```

See the complete scikit-learn documentation about GridSearchCV:

https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html   



2. Randomized Search - randomly test possible combinations of pre-defined hyperparameter values and select the best tested one.

Example:

```py

#import libraries
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

#load the data
iris = load_iris()

#choose the model
logistic = LogisticRegression(solver='saga', tol=1e-2, max_iter=200, random_state=0)

#establish possible hyperparameters
distributions = dict(C=uniform(loc=0, scale=4), penalty=['l2', 'l1'])

#Do a random search in possible combination between the established hyperparameters
clf = RandomizedSearchCV(logistic, distributions, random_state=0)
search = clf.fit(iris.data, iris.target)

#Get the best hyperparameter values
search.best_params_

{'C': 2..., 'penalty': 'l1'}

```

See the complete scikit-learn documentation about RandomizedSearchCV:

https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html   


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

**Examples of questions that hyperparameter tuning will answer for us**

- What should be the value for the maximum depth of the Decision Tree?

- How many trees should I select in a Random Forest model?

- Should use a single layer or multiple layer Neural Network, if multiple layers then how many layers should be there?

- How many neurons should I include in the Neural Network?

- What should be the minimum sample split value for Decision Tree?

- What value should I select for the minimum sample leaf for my Decision Tree?

- How many iterations should I select for Neural Network?

- What should be the value of the learning rate for gradient descent?

- Which solver method is best suited for my Neural Network?

- What is the K in K-nearest Neighbors?

- What should be the value for C and sigma in Support Vector Machine?


Source: 

https://www.geeksforgeeks.org/difference-between-model-parameters-vs-hyperparameters/

https://www.mygreatlearning.com/blog/hyperparameter-tuning-explained/

https://medium.com/codex/do-i-need-to-tune-logistic-regression-hyperparameters-1cb2b81fca69
