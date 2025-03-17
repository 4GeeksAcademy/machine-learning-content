---
description: >-
  Master Support Vector Machines (SVMs) for classification and regression! Learn
  how to maximize margins and use kernels for better model performance.
---
## Support Vector Machine

**Support Vector Machines** (*SVMs*) are a class of supervised learning algorithms widely used in classification and regression. They were introduced in the 1990s and have been an important tool in the field of machine learning ever since.

Imagine a two-dimensional dataset where data from two classes are clearly separable. An SVM will try to find a straight line (or in more general terms, a hyperplane) that separates the two classes. This line is not unique, but the SVM tries to find the one with the largest **margin** between the closest points of the two classes. However, it is not always possible to separate the classes with a linear hyperplane. In these cases, an SVM uses the kernel trick. Essentially, it transforms the input space to a higher dimensional space where the classes become linearly separable.

The **margin** is the distance between the separating hyperplane and the nearest support vectors of each class. An SVM seeks to maximize this margin, since the higher the value, the greater the increase in robustness and generalizability of the model.

![SVM logical](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/svm-logical.PNG?raw=true)

A **kernel** is a function that takes two inputs and transforms them into a single output value. This function is used to transform data that is not linearly separable into a higher dimensional space where it is.

Suppose we have two types of fruit on a table: apples and bananas. If all the apples are on one side and all the bananas are on another, we can easily draw a straight line to separate them. But what if they are mixed, and we can't separate them with a straight line? This is where the kernel comes in: let's imagine we use our hand to gently tap the center of the table, making the fruits jump in the air. While they are in the air (we add a new dimension: height) we could draw a plane (instead of a line) to separate apples and bananas. Then, when the fruits land back on the table, that plane would translate into a curved line or a more complex shape on the table that separates the fruits. The kernel is that hand that makes the fruit jump: it transforms the original data into a space where it can be more easily separated.

SVMs are powerful tools and have been used in a variety of applications, from text classification and image recognition to bioinformatics, including protein classification and detection of genetically predisposed diseases.

### Model hyperparameterization

We can easily build an SVG in Python using the `scikit-learn` library and the `SVC` function. Some of its most important hyperparameters and the first ones we should focus on are:

- `C`: This is the regularization hyperparameter. It controls the trade-off between maximizing margin and minimizing misclassification. A small value for C allows a wider margin at the expense of some misclassification. A high value for C demands correct classification, possibly at the expense of a narrower margin.
- `kernel`: Defines the kernel function to be used in the algorithm. It can be linear, polynomial, RBF, etc.
- `gamma`: Only used for RBF kernel and others. It defines how far the influence of a single training example goes. Low values mean far influence and high values mean near influence.
- `degree`: Used for the polynomial kernel. It is the degree of the polynomial used.

Another very important hyperparameter is the `random_state`, which controls the random generation seed. This attribute is crucial to ensure replicability.

### Using the model in Python

You can easily use `scikit-learn` to program these methods post EDA:

```py
from sklearn.svm import SVC

# Uploading of train and test data
# These data must have been normalized and correctly processed in a complete EDA

model = SVC(kernel = "rbf", C = 1.0, gamma = 0.5)
model.fit(X_train, y_train)

y_pred = model.predict(y_test)
```
