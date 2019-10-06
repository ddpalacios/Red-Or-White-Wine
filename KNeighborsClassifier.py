"""
Here we will implement our own version of the Lazy algorithim.
It is named lazy because it actually does no training at all.
It memorizes the training dataset instead.

it is summarized by the following steps

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
"""

import math
import operator
import random

import numpy as np

"""in order to make predictions, we need to calculate
   the similarity betwwen any two given instances
   This allows us to locate the k most similar data instances
   in the train set for a given member of the test set and make it 
   into a prediction
   """
class KNN(object):
    def __init__(self, k_value):
        self.k = k_value

#Calculate the distance between test data and each row of training data
    def Euclidean(self, test_data, training_data):
        length = test_data.shape[1]
        distance =0
        for i in range(length):
            distance+= np.square(test_data[i] - training_data[i])
        return np.sqrt(distance)


    def fit(self, X_train, X_test):
        distances = {}
        for i in range(len(X_train)):
            dist = self.Euclidean(X_test, X_train[i])
            distances[i] = dist[0]

        #Sort calculated distances based on distance values
        sorted_dist = self.sortDistances(distances)




    def sortDistances(self, distances):
        sorted_dist = sorted(distances.items(), key=operator.itemgetter(1))
        return sorted_dist


    def predict(self, X_test):
        pass










