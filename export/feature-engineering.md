# Feature Engineering

Feature engineering is an essential part of building any intelligent system. This is the reason Data Scientists and Machine learning engineers often spend 70% of their time in the data preparation phase before modeling. 

![density_curve.jpg](../assets/ml_pipeline.jpg)

*A standard machine learning pipeline (source: Practical Machine Learning with Python, Apress/Springer)*

Specifically, we’ll learn how to modify dataset variables to extract meaningful information in order to capture as much insight as possible, leaving datasets and their variables ready to be used in machine learning algorithms.

**What is a feature?**

A feature is typically a specific representation on top of raw data, which is an individual, measurable attribute, typically depicted by a column in a dataset. Considering a generic two-dimensional dataset, each observation is depicted by a row and each feature by a column, which will have a specific value for an observation. That said, a feature is a variable, which we can explain as any characteristic, number, or quantity that can be measured or counted. We call them variables because the values they take may vary.

Examples of variables may be:

Age ( 10,15,17,21,30,...)

Country (USA, Thailand, Japan, Argentina,...)

Energy usage (220, 50, 130, 88,...)

We classify variables in a dataset into one of these major types:

-Numerical variables

-Categorical variables

-Datetime variables

-Mixed variables



Features can be of two major types based on the dataset. Inherent raw features are obtained directly from the dataset with no extra data manipulation or engineering. Derived features are usually obtained from feature engineering, where we extract features from existing data attributes. A simple example would be creating a new feature “BMI” from an employee dataset containing “Weight” and "Height" by just using the formula with weight and height.

**What is feature engineering?**

Feature engineering is the process of using data domain knowledge to create and transform features or variables that make machine learning algorithms work more efficiently. It’s a fundamental task for improving machine learning model performance and prediction accuracy.

Feature engineering can be very time consuming because it includes a number of processes, like:

-Filling missing values within a variable

-Creating or extracting new features from the ones available in your dataset

-Encoding categorical variables into numbers

-Variable transformation



**Why do we need to do feature engineering?**

Every time we start a new machine learning project, whether we receive a raw dataset or we do web scraping to obtain the data, data will most certainly be messy and not suitable for training a model. We need to always perform some data exploration at the beginning to find empty values, outliers, data types, relationships,etc. After we understand better the data we have, we can start doing feature engineering tasks to build high performance models. The succeess of an algorithm can often hinge on how we engineer the input features.

**Do not confuse feature engineering with feature selection**. Feature selection allows us to select features from the feature pool (including any newly-engineered ones) that will help machine learning models make predictions on target variables more efficiently. In a typical machine learning pipeline, we perform feature selection after completing feature engineering.

### Feature Engineering on numeric data

Even though numeric data can be directly fed into machine learning models, we still need to engineer features which are relevant to the problem, before building a model.

A form of raw measures include features which represent frequencies, counts or occurrences of specific attributes. Let’s look at a sample of data from the millionsong dataset which depicts counts or frequencies of songs which have been heard by various users.





### Feature engineering on categorical data







### What are some naive feature engineering techniques that improve model efficacy?

1. Summary statistics (mean, median, mode, min, max, std) for each group of similar records. For example, all female customers between the age 34 and 45 would get their own set of summary statistics.

2. Interactions or ratios between features. For example, var1 * var2.

3. Summaries of features. For example, the number of purchases a customer made in the last 30 days (raw features may be last 10 purchase dates).

4. Splitting feature information manually. For example, customers taller than 1.80cm would be a critical piece of information when recommending car vs SUV.

5. KNN using records in the training set to produce a KNN feature that is fed into another model.

Source:

https://towardsdatascience.com/understanding-feature-engineering-part-1-continuous-numeric-data-da4e47099a7b

https://towardsdatascience.com/understanding-feature-engineering-part-2-categorical-data-f54324193e63

https://medium.com/@mxcsyounes/hands-on-with-feature-engineering-techniques-broad-introduction-def389c1fc25
