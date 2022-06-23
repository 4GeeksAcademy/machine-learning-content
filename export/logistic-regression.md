# Logistic Regression

Logistic Regression is a linear classification algorithm. Classification is a problem in which the task is to assign a category/class to a new instance learning the properties of each class from the existing labeled data, called training set. Examples of classification problems can be classifying emails as spam and non-spam, looking at height, weight, and other attributes to classify a person as fit or unfit, etc.

**Why should we learn Logistic Regression?**

- It is the first supervised learning algorithm that comes to the mind of data science practitioners to create a strong baseline model to check the uplift.

- It is a fundamental, powerful, and easily implementable algorithm. It is also a very intuitive and interpretable model as the final outputs are coefficients depicting the relationship between response variable and features.

**Is logistic regression a regressor or a classifier?**

Logistic regression is usually used as a classifier because it predicts discrete classes.
Having said that, it technically outputs a continuous value associated with each prediction.
So we see that it is actually a regression algorithm that can solve classification problems.

Logistic regression can produce a probability score along with its classification prediction.
It is fair to say that it is a classifier because it is used for classification, although it is technically also a regressor.


## What parameters can be tuned in logistic regression models?







## Using logistic regression in Titanic survival prediction




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

final_df = pd.read_csv('datasets/clean_titanic_train.csv')

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
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>fam_mbrs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>0.271174</td>
      <td>1</td>
      <td>0</td>
      <td>0.027567</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0.472229</td>
      <td>1</td>
      <td>0</td>
      <td>0.271039</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0.321438</td>
      <td>0</td>
      <td>0</td>
      <td>0.030133</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0.434531</td>
      <td>1</td>
      <td>0</td>
      <td>0.201901</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>0.434531</td>
      <td>0</td>
      <td>0</td>
      <td>0.030608</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>0.346569</td>
      <td>0</td>
      <td>0</td>
      <td>0.032161</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0.673285</td>
      <td>0</td>
      <td>0</td>
      <td>0.197196</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>0.019854</td>
      <td>3</td>
      <td>1</td>
      <td>0.080133</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0.334004</td>
      <td>0</td>
      <td>2</td>
      <td>0.042332</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0.170646</td>
      <td>1</td>
      <td>0</td>
      <td>0.114338</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



**Separate features and target as X and y**


```python
X = final_df.drop(['Survived','Unnnamed: 0'], axis=1)
y = final_df['Survived']
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    d:\DANIELA\4Geeks\machine-learning-content\06-12d-ml_algos\logistic-regression.ipynb Cell 10' in <cell line: 1>()
    ----> <a href='vscode-notebook-cell:/d%3A/DANIELA/4Geeks/machine-learning-content/06-12d-ml_algos/logistic-regression.ipynb#ch0000008?line=0'>1</a> X = final_df.drop(['Survived','Unnnamed: 0'], axis=1)
          <a href='vscode-notebook-cell:/d%3A/DANIELA/4Geeks/machine-learning-content/06-12d-ml_algos/logistic-regression.ipynb#ch0000008?line=1'>2</a> y = final_df['Survived']


    File c:\Users\danie\AppData\Local\Programs\Python\Python310\lib\site-packages\pandas\util\_decorators.py:311, in deprecate_nonkeyword_arguments.<locals>.decorate.<locals>.wrapper(*args, **kwargs)
        305 if len(args) > num_allow_args:
        306     warnings.warn(
        307         msg.format(arguments=arguments),
        308         FutureWarning,
        309         stacklevel=stacklevel,
        310     )
    --> 311 return func(*args, **kwargs)


    File c:\Users\danie\AppData\Local\Programs\Python\Python310\lib\site-packages\pandas\core\frame.py:4954, in DataFrame.drop(self, labels, axis, index, columns, level, inplace, errors)
       4806 @deprecate_nonkeyword_arguments(version=None, allowed_args=["self", "labels"])
       4807 def drop(
       4808     self,
       (...)
       4815     errors: str = "raise",
       4816 ):
       4817     """
       4818     Drop specified labels from rows or columns.
       4819 
       (...)
       4952             weight  1.0     0.8
       4953     """
    -> 4954     return super().drop(
       4955         labels=labels,
       4956         axis=axis,
       4957         index=index,
       4958         columns=columns,
       4959         level=level,
       4960         inplace=inplace,
       4961         errors=errors,
       4962     )


    File c:\Users\danie\AppData\Local\Programs\Python\Python310\lib\site-packages\pandas\core\generic.py:4267, in NDFrame.drop(self, labels, axis, index, columns, level, inplace, errors)
       4265 for axis, labels in axes.items():
       4266     if labels is not None:
    -> 4267         obj = obj._drop_axis(labels, axis, level=level, errors=errors)
       4269 if inplace:
       4270     self._update_inplace(obj)


    File c:\Users\danie\AppData\Local\Programs\Python\Python310\lib\site-packages\pandas\core\generic.py:4311, in NDFrame._drop_axis(self, labels, axis, level, errors, consolidate, only_slice)
       4309         new_axis = axis.drop(labels, level=level, errors=errors)
       4310     else:
    -> 4311         new_axis = axis.drop(labels, errors=errors)
       4312     indexer = axis.get_indexer(new_axis)
       4314 # Case for non-unique axis
       4315 else:


    File c:\Users\danie\AppData\Local\Programs\Python\Python310\lib\site-packages\pandas\core\indexes\base.py:6644, in Index.drop(self, labels, errors)
       6642 if mask.any():
       6643     if errors != "ignore":
    -> 6644         raise KeyError(f"{list(labels[mask])} not found in axis")
       6645     indexer = indexer[~mask]
       6646 return self.delete(indexer)


    KeyError: "['Unnnamed: 0'] not found in axis"


**Split dataframe in training set and testing set**


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
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

    c:\Users\danie\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(





    LogisticRegression()




```python
model.coef_
```




    array([[ 2.93931897e-04, -8.75048187e-01, -2.54984177e+00,
            -2.72265416e-02, -9.61048282e-02, -2.07671242e-02,
             1.12536765e+00,  2.30871173e-01, -1.16871952e-01]])






```python
# Hypertune parameters

optimized_model = LogisticRegression(solver='liblinear', penalty='l2', random_state=42, C=0.01)

optimized_model.coef_
```

Source: 

https://towardsdatascience.com/a-handbook-for-logistic-regression-bb2d0dc6d8a8

https://www.displayr.com/how-to-interpret-logistic-regression-coefficients/
