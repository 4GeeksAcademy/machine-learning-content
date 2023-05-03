# Support Vector Machine

Support Vector Machine (SVM) is a supervised learning algorithm so we need to have a labeled dataset to be able to use SVM. It can be used for regression and classification problems and it can be of linear and non linear type. The main objective of SVM is to find a hyperplane in an N( total number of features),dimensional space that differentiates the data points. So we need to find a plane that creates the maximum margin between two data point classes, which means finding the line for which the distance of the closest points is the farthest possible.

Let's see some graphs where the algorithm is trying to find the maximum distance between the closest points:

![svm](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/svm.jpg?raw=true)


![svm2](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/svm2.jpg?raw=true)

![svm3](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/svm3.jpg?raw=true)

We can see that the third graph between lines is the greatest distance we can observe in order to put between our two groups.

To completely understand the Support Vector Machine terminology, let's look at the parts of it:

![svm_terminology](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/svm_terminology.jpg?raw=true)

The margin needs to be equal on both sides. 

To sum up some possible uses of SVM models, they can be used for:

- Linear classification

- Non linear classification

- Linear Regression

- Non linear Regression

## Effect of small margins

The maximum margin classifier will separate the categories. If you are familiar with the bias-variance trade off, you will know that if you are putting a line that is very well adjusted for your dataset, you are reducing the bias and increasing the variance. If you are not familiar yet, what it says is that when you have high bias, it means that your model might not be fully tuned to your dataset so will have more misclassification, but if you have a low bias and high variance it means that your model will be very very well tuned to your dataset and that can lead to overfitting meaning that it will perform very well on your training set but in your testing set you will have more error rate.

Because your margins are so small, you may increase the chance of misclassify new data points.

How do we deal with that?

- Increase margins by including some misclassified data points between your margins. Increase bias and reduce variance can be done by controlling one of the SVM hyperparameters, the 'C' parameter.

A low C value increases bias and decreases variance. If values are too extreme you may underfit.

A high C value decreases bias and increases variance. If values are too extreme you may overfit.

We can determine the best C value by doing cross validation or tuning with validation set.

**What other hyperparameters can be optimized for Support Vector Machine?**

- Kernel: The kernel is decided on the basis of data transformation. By default, the kernel is the Radial Basis Function kernel (RBF). We can change it to linear or Polynomial depending on our dataset.

- C parameter: The c parameter is a regularization parameter that tells the classifier how much misclassification to avoid. If the value of C is high then the classifier will fit training data very well which might cause overfitting. A low C value might allow more misclassification(errors) which can lead to lower accuracy for our classifier.

- Gamma: Gamma is a nonlinear hyperplane parameter. High values indicate that data points that are very close to each other can be grouped. A low value indicates that data points can be grouped together even if they are separated by large distances.

To know the entire list of SVM hyperparameters that can be tuned, go to the following scikit learn documentation: 

https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

**Why is it important to scale features before using SVM?**

SVM tries to fit the widest gap between all classes, so unscaled features can cause some features to have a significantly larger or smaller impact on how the SVM split is created.

**Can SVM produce a probability score along with its classification prediction?**

No.


Source:

https://medium.com/analytics-vidhya/how-to-build-a-simple-sms-spam-filter-with-python-ee777240fc

https://projectgurukul.org/spam-filtering-machine-learning/

https://medium.com/@bedigunjit/simple-guide-to-text-classification-nlp-using-svm-and-naive-bayes-with-python-421db3a72d34

