
# K Nearest neighbors

KNN makes predictions by averaging the k neighbors nearest to a given data point. For example, if we wanted to predict how much money a potential customer would spend at our store, we could find the 5 customers most similar to her and average their spending to make the prediction.

The average could be weighted based on similarity between data points and the similarity distance metric could be defined as well.

**Is KNN a parametric or non-parametric algorithm? Is it used as a classifier or regressor?**

KNN is non-parametric and can be used either as a classifier or regressor.

**How do we select the ideal number of neighbors for KNN?**

There is no closed-form solution for calculating k, so various heuristics are often used. It may be easiest to simply do cross validation and test several different values for k and choose the one that produces the smallest error during cross validation.

As k increases, bias tends to increase and variance decreases.
