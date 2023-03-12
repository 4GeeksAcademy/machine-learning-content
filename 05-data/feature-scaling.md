# Feature scaling

Following the data pre-processing steps, data scaling is a method of standardization that’s useful when working with a dataset that contains continuous features that are on different scales, and you’re using a model that operates in some sort of linear space (like linear regression or K-nearest neighbors).

What does it mean with different scales?

Most of the times, our dataset will contain features highly varying in magnitudes, units and range. But since, most of the machine learning algorithms use Eucledian distance between two data points in their computations, this is a problem. If left alone, these algorithms only take in the magnitude of features neglecting the units. The results would vary greatly between different units, 5kg and 5000gms. The features with high magnitudes will weigh in a lot more in the distance calculations than features with low magnitudes.


So imagine we’re looking at the prices of some products in both Yen and US Dollars. One US Dollar is worth about 100 Yen, but if we don’t scale our prices, algorithms like SVM or KNN will consider a difference in price of 1 Yen as important as a difference of 1 US Dollar!

Remember, scaling means transforming the data so that it fits within a specific scale, like 0-100 or 0-1. Usually 0-1. We want to scale data especially when we’re using methods based on measures of how far apart data points are.

## Methods for scaling the data

We will learn how to implement different scale transformations using in-built functions that come with the scikit-learn package.
Apart from supporting library functions other functions that will be used to achieve the functionality are:

-The fit(data) method is used to compute the mean and std dev for a given feature so that it can be used further for scaling.

-The transform(data) method is used to perform scaling using mean and std dev calculated using the .fit() method.

-The fit_transform() method does both fit and transform.

#### 1. Normalization or scaling

It refers to transforming our input data into a new scale. This distribution will have values between -1 and 1 with μ=0. The normalizer scales each value by dividing each value by its magnitude in n-dimensional space for n number of features.

**Example:**

```py
#import 
from sklearn import preprocessing

#scale the data
scaler = preprocessing.Normalizer()
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, columns=df.columns)

```

**Drawback:** It is sensitive to outliers since the presence of outliers  will compress most values and make them appear extremely close together.

#### 2. Min-Max

Another way of data scaling, which is a linear transformation of data that maps the minimum value to 0 and the maximum value to 1. The minimum of feature is made equal to zero and the maximum of feature equal to one. MinMax Scaler shrinks the data within the given range, usually of 0 to 1. It transforms data by scaling features to a given range. It scales the values to a specific value range without changing the shape of the original distribution.

The MinMax scaling is done using:

```py
x_std = (x – x.min(axis=0)) / (x.max(axis=0) – x.min(axis=0))

x_scaled = x_std * (max – min) + min
```

Where:

-min, max = feature_range

-x.min(axis=0) : Minimum feature value

-x.max(axis=0):Maximum feature value

Sklearn preprocessing defines MinMaxScaler() method to achieve this.

**Example:**

```py

# another way of importing the module
from sklearn.preprocessing import MinMaxScaler
 
# scale features
scaler = MinMaxScaler()
model=scaler.fit(data)
scaled_data=model.transform(data)

```

It essentially shrinks the range such that the range is now between 0 and 1 (or -1 to 1 if there are negative values).

This scaler works better for cases in which the standard scaler might not work so well. If the distribution is not Gaussian or the standard deviation is very small, the min-max scaler works better.

**Drawback:** It is also sensitive to outliers sice the presence will compress most values and make them appear extremely close together.

#### 3. RobustScaler

The RobustScaler uses a similar method to the Min-Max scaler but it instead uses the interquartile range, rathar than the min-max, so that it is robust to outliers. Therefore it follows the following formula for each feature:

```py
xi–Q1(x) / Q3(x)–Q1(x)
```

Of course this means it is using the less of the data for scaling so it’s more suitable for when there are outliers in the data.

**Example:**

```py

#import 
from sklearn import preprocessing

#scale data 

scaler = preprocessing.RobustScaler()
robust_scaled_df = scaler.fit_transform(x)
robust_scaled_df = pd.DataFrame(robust_scaled_df, columns=['x1', 'x2'])

```


#### 4. Standarization (or Z-score transformation)

It transforms each feature to a normal distribution with a mean of 0 and standard deviation of 1. It replaces the values by their z-score.
Standard Scaler helps to get standardized distribution, with a zero mean and standard deviation of one (unit variance). It standardizes features by subtracting the mean value from the feature and then dividing the result by feature standard deviation. 

The standard scaling is calculated as: 

```py
z = (x - u) / s
```

Where:

-z is scaled data.

-x is to be scaled data.

-u is the mean of the training samples

-s is the standard deviation of the training samples.

Sklearn preprocessing supports StandardScaler() method to achieve this directly in merely 2-3 steps.

**Example:**

```py
# import module
from sklearn.preprocessing import StandardScaler
 
# scale data
scaler = StandardScaler()
model = scaler.fit(data)
scaled_data = model.transform(data)

```

**Drawback:** It rescales to an unbounded interval which can be problematic for certain algorithms. For example, some neural networks that expect input values to be inside a specific range.

Standardisation and Mean Normalization can be used for algorithms that assumes zero centric data like Principal Component Analysis(PCA). 

## When should we scale the data? Why?

When our algorithm will weight each input. For example, gradient descent used by many neural nets, or use distance metrics like KNN. 
Model performance can often be improved by normalizing, standarizing, or otherwise scaling the data so that each feature is given relatively equal weight.

It is also important when features are measured in different units. For example, feature A is measured in inches, feature B is measured in feet and feature C is measured in dollars, that they are scaled in a way that they are weighted and/or represented equally.

In some cases, efficacy will not change but perceived feature importance may change. For example, the coefficients in a linear regression.

Scaling our data tipically does not change performance or feature importance for tree-based models since the split points will simply shift to compensate for the scaled data.

>Rule of thumb : any algorithm that computes distance or assumes normality, scale your features!!!

**Some algorithms where feature scaling matters:**

-K-nearest neighbors: should be scaled for all features to weigh in equally.

-Principal Component Analysis 

-Gradient Descent

-Tree based models are not distance based models and can handle varying ranges of features. Scaling is not required while modelling trees.

-Naive Bayes and Linear Discriminant Analysis give weights to the features accordingly, so feature scaling may not have much effect.

Sources:

https://towardsai.net/p/data-science/scaling-vs-normalizing-data-5c3514887a84

https://medium.com/greyatom/why-how-and-when-to-scale-your-features-4b30ab09db5e

https://benalexkeen.com/feature-scaling-with-scikit-learn/

https://www.kaggle.com/code/poonaml/titanic-survival-prediction-end-to-end-ml-pipeline
