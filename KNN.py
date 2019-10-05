import math
import operator
import random

import numpy as np
class knn(object):
    def __init__(self):
        pass
    """in order to make predictions, we need to calculate
    the similarity betwwen any two given instances
    This allows us to locate the k most similar data instances
    in the train set for a given member of the test set and make it 
    into a prediction
    """

    #We want to get the distance measure which is SSR btwn two data arrays
    def euclidean_dist(self, inst1, inst2, length):
        distance =0
        inst1 = inst1.flatten()
        inst2 = inst2.flatten()
        for i in range(length):
            distance += (inst1[i] - inst2[i]) ** 2

        return math.sqrt(distance)

    #after we collected the distance of similaritym we can collect
    #the k most similar instances for a given unseen datapoint
    #We will name this methood get_neighbors rightfully...
    def get_neighbors(self, train_set, test_inst, k):
        distances = np.zeros(train_set.shape[0])
        for i in range(train_set.shape[0]):
            distances[i] = self.euclidean_dist(test_inst, train_set, len(test_inst))
        ind = np.argsort(distances)
        return (ind[0:k])
        # distances = []
        # length = len(test_inst) -1
        #
        # for i in range(len(train_set)):
        #     distance = self.euclidean_dist(test_inst, train_set, length)
        #     distances.append((train_set,distance))
        # distances.sort(key=operator.itemgetter(1))
        # neighbors = []
        # for i in range(k):
        #     neighbors.append(distances[i][0])
        # return neighbors

#Next task is to devise a predicted response based on those neighbors
#we do this by allowing each neighbor to vote for their own class attribute
#and take the majority vote as the prediction
    def get_response(self, votes):
        vote_result = {}
        for key in votes:
            if key in vote_result:
                vote_result[key] += 1
            else:
                vote_result[key] = 1
        final_list = []
        for (number, vote) in vote_result.items():
            if vote == max(vote_result.values()):
                final_list.append(number)
        Winner = random.choice(final_list)
        return (Winner)
        # class_votes = {}
        # for i in range(len(neighbors)):
        #     response = neighbors[i][-1]
        #     response= hash(tuple(np.array(response)))
        #     if response in class_votes:
        #         class_votes[response] +=1
        #     else:
        #         class_votes[response] = 1
        # sorted_votes = sorted(class_votes.items(), key=operator.itemgetter(1), reverse=True)
        # return sorted_votes[0][0]

    def accuracy(self, test_set, predictions):
        correct =0
        for i in range(len(test_set)):
            if test_set[i][-1] is predictions[i]:
                correct +=1
            return (correct/len(test_set))*100



