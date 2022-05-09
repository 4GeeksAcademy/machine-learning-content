
# Feature Engineering

Feature engineering is an essential part of building any intelligent system. This is the reason Data Scientists often spend 70% of their time in the data preparation phase before modeling.

**What is a feature?**

A feature is typically a specific representation on top of raw data, which is an individual, measurable attribute, typically depicted by a column in a dataset. Considering a generic two-dimensional dataset, each observation is depicted by a row and each feature by a column, which will have a specific value for an observation.

Features can be of two major types based on the dataset. Inherent raw features are obtained directly from the dataset with no extra data manipulation or engineering. Derived features are usually obtained from feature engineering, where we extract features from existing data attributes. A simple example would be creating a new feature “BMI” from an employee dataset containing “Weight” and "Height" by just using the formula with weight and height.

### Feature Engineering on numeric data

Even though numeric data can be directly fed into machine learning models, we still need to engineer features which are relevant to the problem 

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
