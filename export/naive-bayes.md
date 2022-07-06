# Naive Bayes

Naive Bayes is a probabilistic algorithm that is simple enough to be completed by hand. This algorithm really makes you understand a big deal about classification. It is probably one of the most useful algorithms for simple Machine Learning use cases.

Do you remember our probability lessons? Naive Bayes Theorem?

Let’s see one example we used to solve with Naive Bayes Theorem.

A doctor knows that meningitis causes stiff neck 50% of the time. The probability of any patient having meningitis is 1/50000, and the probability of any patient having stiff neck is 1/20. If a patient has stiff neck, what’s the probability he/she has meningitis?

A different example would be if we want to predict whether a person will comply to pay taxes based on features like taxable income in dollars and marital status.

**how does this apply to Machine Learning classification problem?**

We are trying to predict something given some conditions. In classification, given a data point X=(x1,x2,…,xn), what are the odds of Y being y. This can be rewritten as the following equation:

P(Y=y |X=(x1,x2...xn))

Applying this equation to our previous example, our Y would be if the person complies to pay taxes, and we would have x1 as taxable income in dollars, and x2 as marital status.

**How does Naive Bayes work?**

The algorithm avoids considering each individual sample in the dataset, like every other machine learning algorithm. Instead, we only need to input the mean, standard deviation, and alfa for the feature belonging to each class. Most of the data is lost in the process because is not even considered. However, if all features are normally distributed, this algorithm can be extremely accurate.

This algorithm NECESSARILY needs you to understand the importance of using distributions in data science. Knowing the best way to model your data, for example, if it belongs to a lognormal distribution or a normal distribution, really makes a difference when building models.

**Types of Naive Bayes algorithm**

There are 3 types of Naïve Bayes algorithm:

- Gaussian Naïve Bayes

- Multinomial Naïve Bayes

- Bernoulli Naïve Bayes

**Applications of Naive Bayes algorithm**

Naive Bayes is one of the most straightforward and fast classification algorithms. It is very well suited for large volumes of data. It is successfully used in various applications such as :

-Spam filtering

-Text classification

-Sentiment analysis

-Recommender systems

It uses the Bayes theorem of probability for the prediction of unknown classes.

## Case Study

We will use a very simple case study from towardsdatascience.com to make Naive Bayes easy to understand. All images from the following example have been taken from Albers Uzila post in towardsdatascience.com.

Let's take the tax evasion data from our previous example, where we needed to predict whether a person will comply to pay taxes based on refund, marital status and taxable income in dollars. We will fit Naive Bayes into train data with 10 observations, then predict a single unseen observation on the test data.

![naivebayes_image1](../assets/naivebayes_image1.jpg)

We have two categorical features (refund and marital status) and one numerical feature (taxable income). Calculating probability will be different for each.

For convenience, we will call refund, marital status, and taxable income x₁, x₂, and x₃, respectively. We will also prefer to divide x₃ by 1000 so we won’t deal with numbers that are too big or too small (for probability). Finally, 'evade' will be y.

Let's look at how Naive Bayes theorem calculates probability for categorical features. In the first image we have the calculation for 'Refund' column, and the second image shows the calculation for 'Marital status' column.

-Calculating conditional probabilities of Refund given Evade:

![naivebayes_image2](../assets/naivebayes_image2.jpg)

-Calculating conditional probabilities of Marital status given Evade:

![naivebayes_image3](../assets/naivebayes_image3.jpg)

For numerical features we need to assume an underlying distribution for each feature. Let’s say you assume x₃ to be normally distributed, now let's calculate probability for x₃ = 80.  μ and σ² represent mean and variance respectively.

![naivebayes_image4](../assets/naivebayes_image4.jpg)

Now we are ready to predict on the unseen information. Check out the solution:


![naivebayes_image5](../assets/naivebayes_image5.jpg)

Example code:

The Naive Bayes classification algorithm’s cannot handle categorical data so we have to convert refund and marital status to numerical format. After separating our data in X and y we can split our data:

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





>Remember, the score of your model does not only depend on the way you tune it, but also on the quality of your data.

For more information on what other hyperparameters can be tuned, go to the following link:
https://scikit-learn.org/stable/modules/naive_bayes.html


Source:

https://towardsdatascience.com/k-nearest-neighbors-naive-bayes-and-decision-tree-in-10-minutes-f8620b25e89b#c71a

https://medium.com/@omairaasim/machine-learning-project-14-naive-bayes-classifier-step-by-step-a1f4a5e5f834

https://www.analyticsvidhya.com/blog/2021/07/performing-sentiment-analysis-with-naive-bayes-classifier/





