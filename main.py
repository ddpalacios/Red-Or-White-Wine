import numpy as np
import matplotlib.pyplot as plt
from dataframe import DataFrame  # our class to view and manipulate our data
from KNeighborsClassifier import KNN

# lets start by opening our dataset and see what columns (Features) we are dealing with...
df = DataFrame('wineanalysis.csv')

# We see an col named Unnamed... lets get rid of it...
df.drop('Unnamed: 0')

# We need to convert our class label (type) into numerical values... this is part of preprocess our data
# There is no convenient function to do this, so we will be using mapping
type_map = {
    'red': 1,
    'white': 0
}
# replaced Red types for 1 and white as 0
df.map_features(type_map, 'type')
df.head()  # All of our data is now numerical

# Selecting features  + target
X, y = df.select_data_points(start=0, end=12, targ=12)

print("Data:\n", X)
print("\nTarget", y)

# splitting our features using scikit learn train-test-split method
# 70 % assigned in X_train & y_train
# 30 % assigned in X_test, y_test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.03, random_state=0, stratify=y)

# because standardization is more optimal with our models, we will use scikit learn to perform
# to standardize our data. We will import the library as so:
from sklearn.preprocessing import StandardScaler

SS = StandardScaler()
X_train_std = SS.fit_transform(X_train)
X_test_std = SS.transform(X_test)

# For our metrics accuracy will be a better option for determining this dataset since we are
# mainly focusing on based on each characteristic, which class will be MORE LIKELY to be white or red wine
# However, we will test both metrics and see what will be their differences

# Our first model we will be testing this data with is AdalineGD (Logistic Regression)
# luckily, we dont have to implement it ourselves, sklearn has it for us as so,

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

lr = LogisticRegression(solver='lbfgs', C=1)
print("\t\t\t\t---USING LOGISTIC REGRESSION TO 'TRAIN' DATA---\n\n")
lr.fit(X_train_std, y_train)
y_pred = lr.predict(X_test_std)

# Now lets use sklearns metric to determine accuracy
accuracy = accuracy_score(y_test, y_pred) * 100
f1 = f1_score(y_test, y_pred, average='weighted') * 100

# 98% accurate... but is it?  FOR BOTH DEFAULT AND TWEAKING Question... How do we know that this accuracy is valid?
# We will further adjust our metrics and models to determine this.


# If the associated labels is NOT equal to what was predicted, SUM up the amount that were misclassified
misclassified = (y_test != y_pred).sum()
print("\nMisclassified: {}".format(misclassified))
print("\t\t\t\t---USING KNN TO 'TRAIN' DATA---\n\n")
knn = KNN(k_value=10)
predictions = []

X_train = X_train[:1000]
y_train = y_train[:1000]
for each_row in range(len(X_test)):
    res, tally = knn.fit(X_train, X_test[each_row], y_train)
    predictions.append(res)
    # print('> predicted=' + repr(res) + 'Tally=' + repr(tally)) + ', actual=' + repr(y_train[]))

Knn_accuracy = knn.getAccuracy(X_test, y_test, predictions)

print("\n\nSklean Logistic Regression:\nAccuracy metric --> {}%\nF1 Score Metric --> {}%\n\nKNN implementation:\n"
      "Accuaracy Metric --> {}%".format(accuracy, f1, Knn_accuracy))

best_score = max(accuracy, f1, Knn_accuracy)
print("\nBest Score from all models and metrics: {}%".format(best_score))
if best_score == accuracy:
    print("From Sklearn Logistic Regression ACCURACY ")

if best_score == f1:
    print("From Sklearn Logistic Regression F1 ")

if best_score == Knn_accuracy:
    print("KNN implementation ACCURACY")
