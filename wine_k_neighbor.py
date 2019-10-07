# Here we will implement our own version of the Lazy algorithim.
# It is named lazy becuase it actually does no training at all.
# It memorizes the training dataset instead.

# it is summerized by the following steps
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

from dataframe import DataFrame  # our class to view and manipulate our data
from KNeighborsClassifier import KNN

df = DataFrame('wineanalysis.csv')

# again, get rid of column name
df.drop('Unnamed: 0')
# and changed our class labels to numeric
df.head()

# We are going to split the data using train_test_split... we just wont use its labels
# it also split for us.
from sklearn.model_selection import train_test_split

X, y = df.select_data_points(start=0, end=12, targ=12)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.03, random_state=0, stratify=y)


knn = KNN(k_value=1)
neighbors, tally, final_vote = knn.fit(X_train, X_test, y_train)

print("Neighbors: {}\nTally: {}\nFinal Vote: {}".format(neighbors, tally, final_vote))
