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

Step 1: Determine K value by Elbow method and specify the number of clusters K

Step 2: Randomly assign each data point to a cluster

Step 3: Determine the cluster centroid coordinates

Step 4: Determine the distances of each data point to the centroids and re-assign each point to the closest cluster centroid based upon minimum distance

Step 5: Calculate cluster centroids again

Step 6: Repeat steps 4 and 5 until we reach global optima where no improvements are possible and no switching of data points from one cluster to other.



Source:

https://becominghuman.ai/comprehending-k-means-and-knn-algorithms-c791be90883d