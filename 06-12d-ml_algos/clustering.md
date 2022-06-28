# Clustering

## K Nearest neighbors

KNN makes predictions by averaging the k neighbors nearest to a given data point. For example, if we wanted to predict how much money a potential customer would spend at our store, we could find the 5 customers most similar to her and average their spending to make the prediction.

The average could be weighted based on similarity between data points and the similarity distance metric could be defined as well.

**Is KNN a parametric or non-parametric algorithm? Is it used as a classifier or regressor?**

KNN is non-parametric and can be used either as a classifier or regressor.

**How do we select the ideal number of neighbors for KNN?**

There is no closed-form solution for calculating k, so various heuristics are often used. It may be easiest to simply do cross validation and test several different values for k and choose the one that produces the smallest error during cross validation.

As k increases, bias tends to increase and variance decreases.

## K-means

K-means clustering is an unsupervised clustering algorithm that partitions observations into k clusters. 

The cluster means are usually randomized at the start (often by choosing random observations from the data) and then updated as more records are observed.

At each iterations, a new observation is assigned to a cluster based on which cluster mean it is nearest and then the means are recalculated, or updated , with the new observation information included.

**What is one common use case for k-mean clustering?**

Customer segmentation is probably the most common use case for k-means clustering (although it has many uses in various industries).

Often, unsupervised clustering is used to identify groups of similar customers or data points, and then another predictive model is trained on each cluster. Then, new customers are first assigned a cluster and then scored using the appropiate model.

**Why is it difficult to identify the ideal number of clusters in a dataset using k-mean clustering?**

There is no ideal number of clusters since increasing the number of clusters always captures more information about the features (the limiting case is k=number of observations, where each observation is a cluster). Having said that, there are various heuristics that attempt to identify the 'optimal' number of clusters by recognizing when increasing the number of clusters only marginally increases the information captured.

The true answer is usually driven by the application, though. If a business has the ability to create four different offers, then they may want to create four customer clusters, regardless of the data.

**What is one heuristic to select 'k' for k-means clustering?**

One such method is the elbow method. It attempts to identify the point at which adding additional clusters only marginally increases the variance explained by the clusters. The elbow is the point at which we begin to see diminishing returns in explained variance when increasing k.

**Does k-means clustering always converge to the same clusters? How does this affect the use of k-means clustering in production models?**

 There is no guarantee that k-means converges to the same set of clusters, even given the same samples from the same population.
 The clusters that are produced may be radically different based on the initial cluster means selected.

 For this reason, it is important that the cluster definitions reamin static when using k-mean clustering in production to ensure that different clusters aren't created each time during training.