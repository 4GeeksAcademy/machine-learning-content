
# Feature scaling

Following the data pre-processing steps, data scaling is a method of standardization that’s useful when working with a dataset that contains continuous features that are on different scales, and you’re using a model that operates in some sort of linear space (like linear regression or K-nearest neighbors).

What does it mean with different scales?

Most of the times, our dataset will contain features highly varying in magnitudes, units and range. But since, most of the machine learning algorithms use Eucledian distance between two data points in their computations, this is a problem. If left alone, these algorithms only take in the magnitude of features neglecting the units. The results would vary greatly between different units, 5kg and 5000gms. The features with high magnitudes will weigh in a lot more in the distance calculations than features with low magnitudes.


So imagine we’re looking at the prices of some products in both Yen and US Dollars. One US Dollar is worth about 100 Yen, but if we don’t scale our prices, algorithms like SVM or KNN will consider a difference in price of 1 Yen as important as a difference of 1 US Dollar!

Remember, scaling means transforming the data so that it fits within a specific scale, like 0-100 or 0-1. Usually 0-1. We want to scale data especially when we’re using methods based on measures of how far apart data points are.

### Methods for scaling the data

1. Normalization or scaling

General terms that refer to transforming our input data into a new scale. This distribution will have values between -1 and 1with μ=0.

**Drawback:** It is sensitive to outliers since the presence of outliers  will compress most values and make them appear extremely close together.

2. Min-Max

Linear transformation of data that maps the minimum value to 0 and the maximum value to 1.

**Drawback:** It is also sensitive to outliers sice the presence will compress most values and make them appear extremely close together.

3. Standarization (or Z-score transformation)

It transforms each feature to a normal distribution with a mean of 0 and standard deviation of 1. It replaces the values by their z-score.
In python sklearn.preprocessing.scale helps us implementing standardisation.

**Drawback:** It rescales to an unbounded interval which can be problematic for certain algorithms. For example, some neural networks that expect input values to be inside a specific range.

Standardisation and Mean Normalization can be used for algorithms that assumes zero centric data like Principal Component Analysis(PCA).

### When should we scale the data? Why?

When our algorithm will weight each input. For example, gradient descent used by many neural nets, or use distance metrics like KNN. 
Model performance can often be improved by normalizing, standarizing, or otherwise scaling the data so that each feature is given relatively equal weight.

It is also important when features are measured in different units. For example, feature A is measured in inches, feature B is measured in feet and feature C is measured in dollars, that they are scaled in a way that they are weighted and/or represented equally.

In some cases, efficacy will not change but perceived feature importance may change. For example, the coefficients in a linear regression.

Scaling our data tipically does not change performance or feature importance for tree-based models since the split points will simply shift to compensate for the scaled data.

**Some algorithms where feature scaling matters:**

-K-nearest neighbors: should be scaled for all features to weigh in equally.

-Principal Component Analysis 

-Gradient Descent

-Tree based models are not distance based models and can handle varying ranges of features. Scaling is not required while modelling trees.

-Naive Bayes and Linear Discriminant Analysis give weights to the features accordingly, so feature scaling may not have much effect.

Sources:

https://towardsai.net/p/data-science/scaling-vs-normalizing-data-5c3514887a84

https://medium.com/greyatom/why-how-and-when-to-scale-your-features-4b30ab09db5e
