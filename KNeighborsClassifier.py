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
import operator
import numpy as np

"""in order to make predictions, we need to calculate
   the similarity between any two given instances
   This allows us to locate the k most similar data instances
   in the train set for a given member of the test set and make it 
   into a prediction
   """


class KNN(object):
    def __init__(self, k_value):
        self.k = k_value
        self.distances = []
        # generate predictions
        self.predictions = []

    def fit(self, X_train, X_test_rows, labels):

        test_row_length = len(X_test_rows) - 1  # Not including lables
        for each_train_row in range(len(X_train)):
            calculated_distance = self.Euclidean(X_test_rows, X_train[each_train_row], test_row_length)
            self.distances.append((X_train[each_train_row], calculated_distance))
        self.distances.sort(key=operator.itemgetter(1))
        neighbors = self.get_neighbors(self.distances)

        final_vote, tally = self.get_response_votes(neighbors, labels)

        return final_vote, tally

        # for row_features in range(X_train.shape[1]):  # for each ROW in the data...
        #     print(X_test[row_features])
        #     print()
        #     calculated_distance = self.Euclidean(X_test[row_features], X_train[row_features], distance)
        #     self.distances[row_features] = calculated_distance[0]
        #
        # # Sort calculated distances based on distance values
        # sorted_dist = self.sortDistances(self.distances)
        #
        # neighbors = self.get_neighbors(sorted_dist)
        #
        # final_vote, tally = self.get_response_votes(neighbors, labels)
        #
        # self.predictions.append(final_vote)
        # return neighbors, tally, final_vote

    def Euclidean(self, test_data, training_data, test_row_length):
        distance = 0
        # Calculate the distance between test data and each row of training data
        for each_elem in range(test_row_length):
            distance += np.square(test_data[each_elem] - training_data[each_elem])
        return np.sqrt(distance)

    def sortDistances(self, distances):
        sorted_dist = sorted(distances.items(), key=operator.itemgetter(1))
        return sorted_dist

    def sortVotes(self, votes):
        sort_vote = sorted(votes.items(), key=operator.itemgetter(1), reverse=True)
        final_vote = sort_vote[0][0]
        tally = sort_vote
        return final_vote, tally

    def get_neighbors(self, sorted_dist):
        neighbors = []
        # Extract top k neighbors from sorted dictionary
        for i in range(self.k):
            neighbors.append(sorted_dist[i][0])
        return neighbors

    def get_response_votes(self, neighbors, train_labels):
        votes = {}
        # Calculate most frequent class in neighbors

        for idx, elem in enumerate(neighbors):
            response = train_labels[idx]  # retrieve labels for X_train set
            print(elem[0], train_labels[idx])
            print()
            if response in votes:
                votes[response] += 1

            else:
                votes[response] = 1

        final_vote, tally = self.sortVotes(votes)
        return final_vote, tally

    def predict(self, X_test):
        pass
