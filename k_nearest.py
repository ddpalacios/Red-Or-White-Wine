#Here we will implement our own version of the Lazy algorithim.
#It is named lazy becuase it actually does no training at all.
#It memorizes the training dataset instead.

#it is summerized by the following steps
'''
1). Choose the number of k and a distance metric.
2). Find the k-nearest neighbors of the sample that we want to classify
3). Assign the class label by the majority vote

Based on the chosen distance metric, the KNN algorithim finds the
k samples in the training dataset that are closest (most similar) to the
point that we want to classify.

The class label of the new data point is then determined by a majority vote
among its k nearest neighbors

The advantage is that it immediatly adapts as we collect new training data

The downside is that the computational complexity grows lineary (O(n))
unless the # of samples has few dimensions (features)

So, to start,  lets gather our dataset once again.
'''













