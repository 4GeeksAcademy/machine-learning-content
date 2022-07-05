# Logistic Regression

Logistic Regression is a linear classification algorithm. Classification is a problem in which the task is to assign a category/class to a new instance learning the properties of each class from the existing labeled data, called training set. Examples of classification problems can be classifying emails as spam and non-spam, looking at height, weight, and other attributes to classify a person as fit or unfit, etc.


In statistics, the logistic model (or logit model) is used to model the probability of a certain class or event existing such as pass/fail, win/lose, alive/dead or healthy/sick.

**Why should we learn Logistic Regression?**

- It is the first supervised learning algorithm that comes to the mind of data science practitioners to create a strong baseline model to check the uplift.

- It is a fundamental, powerful, and easily implementable algorithm. It is also a very intuitive and interpretable model as the final outputs are coefficients depicting the relationship between response variable and features.

**Is logistic regression a regressor or a classifier?**

Logistic regression is usually used as a classifier because it predicts discrete classes.
Having said that, it technically outputs a continuous value associated with each prediction, by producing a probability score along with its classification prediction.

So we see that it is actually a regression algorithm that can solve classification problems.


## What parameters can be tuned in logistic regression models?

There are several hyperparameters that can be optimized in Logistic Regression. To see all the available ones and their possible values, click: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

However, here we will show you the main hyperparameters we may tune in logistic regression.

**Solver** is the algorithm to use in the optimization problem. The choices are {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, and the default value is ’lbfgs’.

- *lbfgs* relatively performs well compared to other methods and it saves a lot of memory, however, sometimes it may have issues with convergence.

- *sag* faster than other solvers for large datasets, when both the number of samples and the number of features are large.

- *saga* the solver of choice for sparse multinomial logistic regression and it’s also suitable for very large datasets.

- *newton-cg* computationally expensive because of the Hessian Matrix.

- *liblinear* recommended when you have a high dimension dataset - solving large-scale classification problems. 

>Note: ‘sag’ and ‘saga’ fast convergence is only guaranteed on features with approximately the same scale. You can preprocess the data with a scaler from sklearn.preprocessing.

**Penalty (or regularization)** intends to reduce model generalization error, and is meant to disincentivize and regulate overfitting. Technique discourages learning a more complex model, so as to avoid the risk of overfitting. The choices are: {‘l1’, ‘l2’, ‘elasticnet’, ‘none’}, and the default value is ’l2’. 

>Note: not all solvers support all regularization terms.

**C (or regularization strength)** must be a positive float. Regularization strength works with the penalty to regulate overfitting. Smaller values specify stronger regularization and high value tells the model to give high weight to the training data.   


Logistic regression offers other parameters like: 

-class_weight

-dualbool (for sparse datasets when n_samples > n_features)

-max_iter (may improve convergence with higher iterations), and others. However, these provide less impact.

## Using logistic regression in Titanic survival prediction

In Titanic dataset we have more than one feature, and with logistic regression we predict whether they survive or not. If the value the model predict would be 0.85, that would mean the person is 85% a survivor and 15% not a survivor.

Let's imagine the titanic dataset only had one feature(Sex) and we would like to predict if the passenger survives or not, what we are trying to calculate is basically:

$P(survive|sex=male)$

$P(survive|sex=female)$

When the probability is greater or equal than 0.5, the binary value is 1, and when the probability is less than 0.5, the binary value is 0. So, the person we just mentioned above would be classified as 1, which means he or she survived. 

### The logistic regression curve

![logistic_regression_curve](../assets/logistic_regression_curve.jpg)

The logistic function, also called the sigmoid function is an S-shaped curve that can take any real-valued number and map it into a value between 0 and 1, but never exactly at those limits.

Logistic regression uses maximum-likelihood estimation, which calculates the likelihood of one person being a survivor on Titanic, and then another one, and then another one. After all calculations are done, the model multiplies all those likelihoods and fits the S-shaped line to the data. Keeps calculating, until it finds the best S-shaped line. 

### The logistic regression equation

Input values (x) are combined linearly using weights or coefficient values (referred to as the Greek capital letter Beta) to predict an output value (y). A key difference from linear regression is that the output value being modeled is a binary values (0 or 1) rather than a numeric value.

Below is an example logistic regression equation:

$y = e^(b0 + b1*x) / (1 + e^(b0 + b1*x))$

Where y is the predicted output, b0 is the bias or intercept term and b1 is the coefficient for the single input value (x). Each column in your input data has an associated b coefficient (a constant real value) that must be learned from your training data. This is done using maximum-likelihood estimation. The intuition for maximum-likelihood for logistic regression is that a search procedure seeks values for the coefficients (Beta values) that minimize the error in the probabilities predicted by the model to those in the data.

The actual representation of the model that you would store in memory or in a file are the coefficients in the equation (the beta value or b’s).

To get the probabilities with Python you can use:

```py

y_pred = model.predict_proba(X_test)
roc_auc_score(y_test, y_pred)

```

This will give you an array of 'probability to be in each class' given to each observation, but you can also get the probability to be in the class '1':

```py

# Score

y_pred = model.predict_proba(X_test)[:, 1]
roc_auc_score(y_test, y_pred)

```

### Titanic Modeling


```python
# import libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

**Loading the final dataframe**

In order to start modeling, we will focus on our train data and forget temporarily about the dataset where we need to make predictions.

Let's start by loading our clean titanic train data and name it final_df.


```python
# loading clean train dataset

final_df = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/machine-learning-content/master/assets/clean_titanic_train.csv')
```


```python
# Let's take a look at our first 10 rows to verify all data is numerical

final_df.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Embarked_C</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
      <th>Female</th>
      <th>Male</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>fam_mbrs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.271174</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.027567</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.472229</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.271039</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.321438</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.030133</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.434531</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.201901</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.434531</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.030608</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.346569</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.032161</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.673285</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.197196</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.019854</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>0.080133</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.334004</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.042332</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.170646</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.114338</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



**Separate features and target as X and y**


```python
X = final_df.drop(['Survived','Unnamed: 0'], axis=1)
y = final_df['Survived']
```

**Split dataframe in training set and testing set**


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```


```python
from sklearn.linear_model import LogisticRegression

# Instantiate Logistic Regression

model = LogisticRegression()
```


```python
# Fit the data

model.fit(X_train, y_train)
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression()</pre></div></div></div></div></div>




```python
# Make predictions
y_pred = model.predict(X_test)
y_pred
```




    array([0., 1., 1., 0., 1., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1.,
           0., 1., 1., 0., 0., 0., 0., 1., 0., 1., 0., 1., 1., 1., 0., 1., 1.,
           1., 0., 1., 1., 1., 1., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0.,
           0., 1., 0., 1., 1., 1., 0., 0., 1., 0., 0., 0., 0., 0., 1., 1., 0.,
           0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 1., 0.,
           1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 1., 1., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0., 0., 1., 0.,
           1., 1., 1., 1., 0., 0., 1., 0., 1., 1., 0., 1., 0., 1., 0., 1., 1.,
           0., 0., 1., 1., 1., 0., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 1.,
           1., 1., 1., 0., 1., 0., 1., 1.])




```python
# Check the accuracy score

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_auc_score

accuracy_score(y_test, y_pred)
```




    0.8146067415730337




```python
# Confusion Matrix

titanic_cm = confusion_matrix(y_pred, y_test)
titanic_cm
```




    array([[87, 16],
           [17, 58]])



If you want you can put your confusion matrix a more beautiful graph


```python
# convert the ndarray to a pandas dataframe
cm_df = pd.DataFrame(titanic_cm)
# set the size of the figure
plt.figure(figsize = (5,5))
sns.heatmap(cm_df, 
           annot=True, annot_kws={"size": 25},
           fmt="d",         # decimals format
           xticklabels=False, 
           yticklabels=False,
           cmap="viridis", 
           cbar=False)
plt.show()
```


    
![png](logistic-regression_files/logistic-regression_20_0.png)
    


Observations: 16 + 17 = 33 wrong prediction. Can we can do better?

Let's use GridSearch to see what would be the best hyperparameters for our Logistic Regression model. 

The code below demonstrates grid searching the key hyperparameters for LogisticRegression on a binary classification dataset, in this case our Titanic dataset.


```python
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# define models and parameters

model = LogisticRegression()
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]

# define grid search

grid = dict(solver=solvers,penalty=penalty,C=c_values)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X, y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
```

    Best: 0.798102 using {'C': 0.1, 'penalty': 'l2', 'solver': 'newton-cg'}
    0.793216 (0.032933) with: {'C': 100, 'penalty': 'l2', 'solver': 'newton-cg'}
    0.793216 (0.032933) with: {'C': 100, 'penalty': 'l2', 'solver': 'lbfgs'}
    0.792833 (0.031946) with: {'C': 100, 'penalty': 'l2', 'solver': 'liblinear'}
    0.793212 (0.032824) with: {'C': 10, 'penalty': 'l2', 'solver': 'newton-cg'}
    0.793212 (0.032824) with: {'C': 10, 'penalty': 'l2', 'solver': 'lbfgs'}
    0.793969 (0.031699) with: {'C': 10, 'penalty': 'l2', 'solver': 'liblinear'}
    0.796582 (0.034081) with: {'C': 1.0, 'penalty': 'l2', 'solver': 'newton-cg'}
    0.796582 (0.034081) with: {'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs'}
    0.796229 (0.036167) with: {'C': 1.0, 'penalty': 'l2', 'solver': 'liblinear'}
    0.798102 (0.037820) with: {'C': 0.1, 'penalty': 'l2', 'solver': 'newton-cg'}
    0.798102 (0.037820) with: {'C': 0.1, 'penalty': 'l2', 'solver': 'lbfgs'}
    0.797706 (0.040030) with: {'C': 0.1, 'penalty': 'l2', 'solver': 'liblinear'}
    0.783801 (0.034749) with: {'C': 0.01, 'penalty': 'l2', 'solver': 'newton-cg'}
    0.783801 (0.034749) with: {'C': 0.01, 'penalty': 'l2', 'solver': 'lbfgs'}
    0.783048 (0.033317) with: {'C': 0.01, 'penalty': 'l2', 'solver': 'liblinear'}


Solver newton-cg supports only 'l2' or 'none' penalties, that is why we only included 'l2'.


Observations: We actually got a better score first. Let's confirm these by getting the confusion matrix again.


```python
# Hypertune parameters

optimized_model = LogisticRegression(C= 0.1, penalty='l2', solver= 'newton-cg')

```


```python
# Fit the data

optimized_model.fit(X_train, y_train)
```




<style>#sk-container-id-3 {color: black;background-color: white;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression(C=0.1, solver=&#x27;newton-cg&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" checked><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression(C=0.1, solver=&#x27;newton-cg&#x27;)</pre></div></div></div></div></div>




```python
# Make predictions
y_pred = optimized_model.predict(X_test)
y_pred
```




    array([0., 1., 1., 0., 1., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1.,
           0., 1., 1., 0., 0., 0., 0., 1., 0., 1., 0., 1., 1., 1., 0., 1., 1.,
           1., 0., 1., 1., 1., 1., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0.,
           0., 1., 0., 1., 1., 1., 1., 0., 1., 0., 0., 0., 1., 0., 1., 1., 0.,
           0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 1., 1., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0., 0., 1., 0.,
           1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 0., 1., 0., 1., 0., 1., 1.,
           0., 0., 1., 1., 1., 0., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 1.,
           1., 1., 1., 0., 1., 0., 1., 1.])




```python
# Check the accuracy score

from sklearn.metrics import accuracy_score, confusion_matrix

accuracy_score(y_pred, y_test)
```




    0.8089887640449438




```python
# Confusion Matrix

confusion_matrix(y_pred, y_test)
```




    array([[85, 15],
           [19, 59]])



Observations: In effect, we now have 34 (15 + 19) wrong predictions. We had 33 with our first baseline model.

In the last data preprocessing module we learned some techniques of feature selection if needed. However, the final clean dataset was saved before any feature selection implementation. So let's a feature selection technique here and see if it makes our model improve.

**Recursive feature elimination**

Given an external estimator that assigns weights to features, recursive feature elimination (RFE) is to select features by recursively considering smaller and smaller sets of features. First, the estimator is trained on the initial set of features and the importance of each feature is obtained either through a coef_ attribute or through a feature_importances_ attribute. Then, the least important features are pruned from current set of features.That procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached.


```python
from sklearn.feature_selection import RFE

# create the RFE model and select 9 attributes
rfe = RFE(model, n_features_to_select=9)
rfe = rfe.fit(X, y)

# summarize the selection of the attributes
print('Selected features: %s' % list(X.columns[rfe.support_]))
```

    Selected features: ['Embarked_C', 'Embarked_Q', 'Embarked_S', 'Female', 'Male', 'Pclass', 'Age', 'SibSp', 'fam_mbrs']


These are the 9 attributes suggested. But is 9 the optimal number of Titanic features to enter my model?

**Feature ranking with recursive feature elimination and cross-validation**

RFECV performs RFE in a cross-validation loop to find the optimal number or the best number of features. Hereafter a recursive feature elimination applied on logistic regression with automatic tuning of the number of features selected with cross-validation.


```python
from sklearn.feature_selection import RFECV

# Create the RFE object and compute a cross-validated score.
# The "accuracy" scoring is proportional to the number of correct classifications

rfecv = RFECV(estimator=LogisticRegression(), step=1, cv=10, scoring='accuracy')
rfecv.fit(X, y)

print("Optimal number of features: %d" % rfecv.n_features_)
print('Selected features: %s' % list(X.columns[rfecv.support_]))

# Plot number of features VS. cross-validation scores

plt.figure(figsize=(10,6))
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
```

    Optimal number of features: 8
    Selected features: ['Embarked_C', 'Embarked_Q', 'Embarked_S', 'Female', 'Male', 'Pclass', 'Age', 'SibSp']


    /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages/sklearn/utils/deprecation.py:103: FutureWarning: The `grid_scores_` attribute is deprecated in version 1.0 in favor of `cv_results_` and will be removed in version 1.2.
      warnings.warn(msg, category=FutureWarning)



    
![png](logistic-regression_files/logistic-regression_34_2.png)
    


Observations: We see that the optimal number of features suggested is actually 8. The total number of family members was not considered as important.

**Model evaluation based on K-fold cross-validation using cross_val_score() function**

- This time we are going to use a new features dataframe (new_X) with only the selected features.

- This time the split method will be K-fold cross-validation with 10 folds.

- The model will still be Logistic regression but we will use the best hyperparameters we got in the gridsearch.

- We know that accuracy is not always the best or always poor measure for assesing. There are many metrics like Recall, Precision, F1 Score and much more. We will propose 3 different evaluation metrics. The metrics proposed will be 'accuracy', 'neg_log_loss', and 'roc_auc'.

- Finally, the cross_val_score() function will be used to perform the evaluation, taking the dataset and cross-validation configuration and returning a list of scores calculated for each fold.



```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score 
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss

Selected_features = ['Embarked_C', 'Embarked_Q', 'Embarked_S', 'Female', 'Male', 'Pclass', 'Age', 'SibSp']

# create new X (selected features) 
new_X = final_df[Selected_features]

# 10-fold cross-validation logistic regression, using the best hyperparameters gridsearch got for us.

logreg = LogisticRegression(C= 0.1, penalty='l2', solver= 'newton-cg')

# Use cross_val_score function
# We are passing the entirety of X and y, not X_train or y_train anymore.
# cv=10 for 10 folds
# scoring = {'accuracy', 'neg_log_loss', 'roc_auc'} for evaluation metric

scores_accuracy = cross_val_score(logreg, new_X, y, cv=10, scoring='accuracy')
scores_log_loss = cross_val_score(logreg, new_X, y, cv=10, scoring='neg_log_loss')
scores_auc = cross_val_score(logreg, new_X, y, cv=10, scoring='roc_auc')

print('K-fold cross-validation results:')
print(logreg.__class__.__name__+" average accuracy is %2.3f" % scores_accuracy.mean())
print(logreg.__class__.__name__+" average log_loss is %2.3f" % -scores_log_loss.mean())
print(logreg.__class__.__name__+" average auc is %2.3f" % scores_auc.mean())
```

    K-fold cross-validation results:
    LogisticRegression average accuracy is 0.798
    LogisticRegression average log_loss is 0.459
    LogisticRegression average auc is 0.847


Notice that by using the 'roc_auc' metric we would have got a better score than accuracy. If we would want to repeat our modeling process using this evaluation metric, consider using the predict_proba(X_test) to predict the probabilities also graph auc_roc curves, because if you use model.predict(X_test), the method won’t have all the necessary information to build all the points in the curve.

Let's finish our class predictions!

We need to fit the whole training data to the estimator once we are satisfied with the results of cross_val_score, before we can use it to predict on Titanic test data.


```python
# Let's fit our entire titanic train data with this model

logreg.fit(new_X,y)
```




<style>#sk-container-id-5 {color: black;background-color: white;}#sk-container-id-5 pre{padding: 0;}#sk-container-id-5 div.sk-toggleable {background-color: white;}#sk-container-id-5 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-5 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-5 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-5 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-5 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-5 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-5 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-5 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-5 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-5 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-5 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-5 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-5 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-5 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-5 div.sk-item {position: relative;z-index: 1;}#sk-container-id-5 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-5 div.sk-item::before, #sk-container-id-5 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-5 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-5 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-5 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-5 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-5 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-5 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-5 div.sk-label-container {text-align: center;}#sk-container-id-5 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-5 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-5" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression(C=0.1, solver=&#x27;newton-cg&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" checked><label for="sk-estimator-id-5" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression(C=0.1, solver=&#x27;newton-cg&#x27;)</pre></div></div></div></div></div>



In the following code, let's see how to generate our predictions, and save them in a predictions file with the right Id.


```python
#Load initial test set (the one that was not cleaned yet and where we need to make predictions)

initial_test_df = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/machine-learning-content/master/assets/titanic_test.csv')

#Load cleaned test set

final_test = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/machine-learning-content/master/assets/clean_titanic_test.csv')

#Adding the column survived to clean test set in order to make predictions

final_test['Survived'] = logreg.predict(final_test[Selected_features])

#Putting the 'PassengerId' column back to test set

final_test['PassengerId'] = initial_test_df['PassengerId']

#Creating a final predictions dataframe

submission = final_test[['PassengerId','Survived']]

# Convert final dataframe in a csv file
# Normally this is the required csv file in Kaggle competitions

submission.to_csv("submission.csv", index=False)
```

Take a look at our predictions in the last rows of the test set


```python
submission.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>413</th>
      <td>1305</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>414</th>
      <td>1306</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>415</th>
      <td>1307</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>416</th>
      <td>1308</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>417</th>
      <td>1309</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



## Important considerations before modeling with Logistic Regression 

Ultimately in predictive modeling machine learning projects we are focused on making accurate predictions rather than interpreting the results. As such, we can break some assumptions as long as the model is robust and performs well.

- Binary Output Variable: This might be obvious as we have already mentioned it, but logistic regression is intended for binary (two-class) classification problems. It will predict the probability of an instance belonging to the default class, which can be snapped into a 0 or 1 classification.

- Remove Noise: Logistic regression assumes no error in the output variable (y), consider removing outliers and possibly misclassified instances from your training data.

- Gaussian Distribution: Logistic regression is a linear algorithm (with a non-linear transform on output). It does assume a linear relationship between the input variables with the output. Data transforms of your input variables that better expose this linear relationship can result in a more accurate model. For example, you can use log, root, Box-Cox and other univariate transforms to better expose this relationship.

- Remove Correlated Inputs: Like linear regression, the model can overfit if you have multiple highly-correlated inputs. Consider calculating the pairwise correlations between all inputs and removing highly correlated inputs.

- Fail to Converge: It is possible for the expected likelihood estimation process that learns the coefficients to fail to converge. This can happen if there are many highly correlated inputs in your data or the data is very sparse (e.g. lots of zeros in your input data).

Considering this, are there any changes you would do in your data preprocessing phase?

Source: 

https://towardsdatascience.com/a-handbook-for-logistic-regression-bb2d0dc6d8a8

https://www.displayr.com/how-to-interpret-logistic-regression-coefficients/

https://towardsdatascience.com/a-handbook-for-logistic-regression-bb2d0dc6d8a8

https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/#:~:text=Logistic%20regression%20does%20not%20really,with%20different%20solvers%20(solver).&text=Regularization%20(penalty)%20can%20sometimes%20be%20helpful.

https://machinelearningmastery.com/logistic-regression-for-machine-learning/

https://www.kaggle.com/code/mnassrib/titanic-logistic-regression-with-python/notebook

https://www.kaggle.com/code/rmiperrier/tps-mar-lgbm-predict-proba-vs-predict?scriptVersionId=55643096

https://medium.com/codex/do-i-need-to-tune-logistic-regression-hyperparameters-1cb2b81fca69

