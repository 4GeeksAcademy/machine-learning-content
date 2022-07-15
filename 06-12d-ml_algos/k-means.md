# K-means

Unsupervised learning models are used when we only have the input variables (X) and no corresponding output variables. They use unlabeled training data to model the underlying structure of the data.

Clustering is used to group samples such that objects within the same cluster are more similar to each other than to the objects from another cluster.

K-means clustering is an unsupervised clustering algorithm that takes a bunch of unlabeled points and tries to group them into “k” number of clusters.

The “k” in k-means denotes the number of clusters you want to have in the end. If k = 5, you will have 5 clusters on the data set.

The cluster means are usually randomized at the start (often by choosing random observations from the data) and then updated as more records are observed.

At each iterations, a new observation is assigned to a cluster based on which cluster mean it is nearest and then the means are recalculated, or updated , with the new observation information included.

**What is one common use case for k-mean clustering?**

Customer segmentation is probably the most common use case for k-means clustering (although it has many uses).

## How does it work?





Source:

https://becominghuman.ai/comprehending-k-means-and-knn-algorithms-c791be90883d