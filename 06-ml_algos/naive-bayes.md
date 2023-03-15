# Naive Bayes

Naive Bayes is a probabilistic algorithm for classification, based on Bayes Theorem. This algorithm really makes you understand a big deal about classification and it is probably one of the most useful algorithms for simple Machine Learning use cases. It is simple enough to be completed by hand.

Do you remember our probability lessons? Naive Bayes Theorem?

Let’s see one example we used to solve with Naive Bayes Theorem.

A doctor knows that meningitis causes stiff neck 50% of the time. The probability of any patient having meningitis is 1/50000, and the probability of any patient having stiff neck is 1/20. If a patient has stiff neck, what’s the probability he/she has meningitis?

A different example would be if we wanted to predict whether a person will comply to pay taxes based on features like taxable income and marital status.

**how does this apply to Machine Learning classification problem?**

We are trying to predict something given some conditions. In classification, given a data point X=(x1,x2,…,xn), what are the odds of Y being y. This can be rewritten as the following equation:

P(Y=y |X=(x1,x2...xn))

Applying this equation to our previous example, our Y would be if the person complies to pay taxes, and we would have x1 as taxable income and x2 as marital status.

## How does Naive Bayes work?

The algorithm avoids considering each individual sample in the dataset, like every other machine learning algorithm. Instead, we only need to input the mean, standard deviation, and alfa for the feature belonging to each class. Most of the data is lost in the process because is not even considered. However, if all features are normally distributed, this algorithm can be extremely accurate.

This algorithm NECESSARILY needs you to understand the importance of using distributions in data science. Knowing the best way to model your data, for example, if it belongs to a lognormal distribution or a normal distribution, really makes a difference when building models.

### Advantages and disadvantages of Naïve Bayes algorithm:

**Advantages:**

- Naive Bayes is a really unique machine learning algorithm because it doesn’t learn through gradient descent mechanism. Instead Naive Bayes calculates its own parameters in a very fast calculation.

- This causes Naive Bayes to be an astonishingly fast machine learning algorithm compared to most of its ML algorithm competition which learn through iterative gradient descent process or distance calculations.

- Naive Bayes is commonly used and preferred in suitable classification tasks. It can be used for binary and multiclass classification.

- Naïve Bayes have very fast training and prediction phases. Because of its high performance Naive Bayes can be preferred in real time machine learning deployment or as a complimentary model to improve less speedy models in a hybrid solution.

- It is a great choice for text classification problems. It is a popular choice for spam email classification.

- It can be easily trained on small dataset. Some machine learning models will require lots of data for training and perform poorly if it’s not provided. Naive Bayes usually performs very well with moderate sizes of training datasets.

- It will produce probability reports which can be very useful if you need them. They don’t just tell target label of a sample but they also tell the probability of the prediction, which gives you increased control on the classification process by allowing or disallowing predictions below a certain probability percentage.

**Disadvantages:**

- Naïve Bayes algorithm involves the use of the Bayes theorem. So, it does not work well when we have particular missing values or missing combination of values.

- Naïve Bayes algorithm works well when we have simple categories. But, it does not work well when the relationship between words is important.

### Types of Naive Bayes algorithm

There are 3 types of Naïve Bayes algorithm:

- Gaussian Naïve Bayes

- Multinomial Naïve Bayes

- Bernoulli Naïve Bayes

While Gaussian Naive Bayes can handle continuous data, Bernoulli Naive Bayes works great with binary and Multinomial Naive Bayes can be used to classify categorical discrete multiclass datasets.

This offers great versatility and enables many advantages for text related classification implementations. There is even Complement Naive Bayes which is a variant of Multinomial Naive Bayes and it aims to improve model accuracy when working with imbalanced data by reducing bias through a complement approach.

### Applications of Naive Bayes algorithm

Naive Bayes is one of the most straightforward and fast classification algorithms. It is very well suited for large volumes of data. It is successfully used in various applications such as :

- Spam filtering

- Text classification

- Sentiment analysis

- Recommender systems

It uses the Bayes theorem of probability for the prediction of unknown classes. Its assumption of feature independence, and its effectiveness in solving multi-class problems, makes it perfect for performing Sentiment Analysis. Sentiment Analysis refers to the identification of positive or negative sentiments of a target group (customers, audience, etc.)

## Case Study

We will use a very simple case study from towardsdatascience.com to make Naive Bayes easy to understand. All images from the following example have been taken from Albers Uzila post in towardsdatascience.com.

Let's take the tax evasion data from our previous example, where we needed to predict whether a person will comply to pay taxes based on marital status and taxable income. We will add a new predictor 'refund' and fit Naive Bayes into train data with 10 observations, then predict a single unseen observation on the test data.

![naivebayes_image1](../assets/naivebayes_image1.jpg)

We now have two categorical features (refund and marital status) and one numerical feature (taxable income). Calculating probability will be different for each.

For convenience, we will call refund, marital status, and taxable income x₁, x₂, and x₃, respectively. We will also prefer to divide x₃ by 1000 so we won’t deal with numbers that are too big or too small (for probability). Finally, 'evade' will be y.

Let's look at how Naive Bayes theorem calculates probability for categorical features. In the first image we have the calculation for 'Refund' column, and the second image shows the calculation for 'Marital status' column.

- Calculating conditional probabilities of Refund given Evade:

![naivebayes_image2](../assets/naivebayes_image2.jpg)

- Calculating conditional probabilities of Marital status given Evade:

![naivebayes_image3](../assets/naivebayes_image3.jpg)

For numerical features we need to assume an underlying distribution for each feature. Let’s say you assume x₃ to be normally distributed, now let's calculate probability for x₃ = 80.  μ and σ² represent mean and variance respectively.

![naivebayes_image4](../assets/naivebayes_image4.jpg)

Now we are ready to predict on the unseen information. Check out the solution:


![naivebayes_image5](../assets/naivebayes_image5.jpg)

Example code:

The Naive Bayes classification algorithm’s cannot handle categorical data so we have to convert refund and marital status to numerical format. After categorical encoding and separating our data in X (features) and y(target) we can split our data for modeling:

```py

# Splitting the data into Train and Test

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Step 4 - Feature Scaling

from sklearn.preprocessing import StandardScaler
ss_X = StandardScaler()
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)

# Step 5 - Fit the classifier

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Step 6 - Predict

y_pred = classifier.predict(X_test)

# Step 7 - Confusion Matrix

from sklearn import metrics
cm = metrics.confusion_matrix(y_test, y_pred) 
print(cm)
accuracy = metrics.accuracy_score(y_test, y_pred) 
print("Accuracy score:",accuracy)
precision = metrics.precision_score(y_test, y_pred) 
print("Precision score:",precision)
recall = metrics.recall_score(y_test, y_pred) 
print("Recall score:",recall)
```

## Hypertuning Naive Bayes

Naive Bayes model has a couple of useful hyperparameters to tune in Scikit-Learn. Aside of hyperparameters probably the most importatant factor in a Naive Bayes implementation is the independence of predictors (features).

This machine learning model is called Naive because it assumes independence between features which is rarely the case hence the model is being naive about it.

Here are some two useful parameters that can be optimized and tuned in Naive Bayes:

- **Priors:**

Priors are probabilities of a feature or target before new data is introduced to the model. Priors parameter gives an option to specify priors instead of model deriving them from the frequencies in data. Sum of priors should always add up to 1. This is true when model is calculating the priors as well as when user is passing priors as an array to the model. If priors don’t add up to 1 following error will likely occur.

```py
GNB = GaussianNB(priors = [0.33, 0.33, 0.34])
```

Adjusting priors can be useful to address bias in a dataset. For example if dataset is small and target values are occuring in a biased way Naive Bayes model may think that the frequency of target A is less than target B and C and this will affect the results. But by intercepting and assigning custom priors that you know are more accurate you can contribute to the accuracy of the model.

- **var_smoothing:**

Var_smoothing is a parameter that takes a float value and is 1e-9 by default. It is a stability calculation to widen (or smooth) the curve and therefore account for more samples that are further away from the distribution mean.

```py
GNB = GaussianNB(var_smoothing=3e-10)
```

If var_smoothing is increased too much likelihood probability for all classes will converge to a uniform distribution. Meaning prediction will be distributed to target values at equal probability which renders predictions pretty much useless and turns Naive Bayes machine learning model to a coin toss.

Benefit of var_smoothing is that when there is missing data or a class is not represented var_smoothing keeps model from breaking down.

Thanks to adding var_smoothing to prior, model can end up with a more reasonable equation instead of being confused when an event has never occurred before. And since its usually a very small value when priors are present var_smoothing’s affect is negligible. 

Laplace is the father of this smoothing and he came up with the idea when he was thinking about the probability of sun not rising up. He thought since it never happened before Bayes Theorem couldn’t deal with it. 

It could be tweaked to assign it a probability of 1 which makes no sense (100% chance) or it could also be tweaked to assign the event a probability of 0. Which is probably more sensible but still not ideal since we can’t say it’s improbable just because it has never happened before. Financial markets taught us that time and again. Hence introducing a tiny variance for those previously unseen observations we save the model’s integrity.

>Remember, the score of your model does not only depend on the way you tune it, but also on the quality of your data.

**Tips to improve the power of Naïve Bayes Model**

- If continuous features do not have normal distribution, we should use transformation or different methods to convert it in normal distribution.

- If test data set has zero frequency issue, apply smoothing techniques “Laplace Correction” to predict the class of test data set. Laplacian correction is one of the smoothing techniques. Here, you can assume that the dataset is large enough that adding one row of each class will not make a difference in the estimated probability. This will overcome the issue of probability values to zero.

- Remove correlated features, as the highly correlated features are voted twice in the model and it can lead to over inflating importance.

- Naïve Bayes classifiers have limited options for parameter tuning like alpha=1 for smoothing, fit_prior=[True / False] to learn class prior probabilities or not and some other options (look at detail here: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB). It's recommended to focus on your pre-processing of data and the feature selection.

- You might think to apply some classifier combination technique like ensemble, bagging and boosting but these methods would not help. Actually, “ensemble, boosting, bagging” won’t help since their purpose is to reduce variance. Naïve Bayes has no variance to minimize.


Source:

https://towardsdatascience.com/k-nearest-neighbors-naive-bayes-and-decision-tree-in-10-minutes-f8620b25e89b#c71a

https://medium.com/@omairaasim/machine-learning-project-14-naive-bayes-classifier-step-by-step-a1f4a5e5f834

https://www.analyticsvidhya.com/blog/2021/07/performing-sentiment-analysis-with-naive-bayes-classifier/

https://deeppatel23.medium.com/na%C3%AFve-bayes-classifier-177ad1307aff

https://github.com/pb111/Naive-Bayes-Classification-Project

https://www.analyticsvidhya.com/blog/2021/01/gaussian-naive-bayes-with-hyperpameter-tuning/

https://www.aifinesse.com/naive-bayes/naive-bayes-advantages/

https://www.aifinesse.com/naive-bayes/naive-bayes-tuning/




