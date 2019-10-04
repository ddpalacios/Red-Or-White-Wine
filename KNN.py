import math
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


