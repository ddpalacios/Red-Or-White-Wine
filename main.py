import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dataframe import DataFrame  # our class to view and manipulate our data

# lets start by opening our dataset and see what columns (Features) we are dealing with...
df = DataFrame('wineanalysis.csv')
data_frame = df.df

# We see an col named Unnamed... lets get rid of it...
df.drop('Unnamed: 0')

# We need to convert our class lable (type) into numerical values... this is part of preprocessing our data
# There is no convenient function to do this, so we will be using mapping
type_map = {
    'red': 1,
    'white': 0
}
# replaced Red types for 1 and white as 0
df.map_features(type_map, 'type')
df.head()  # All of our data is now numerical

# Selecting feautures  + target
X, y = df.select_data_points(12, 12)

print("Data:\n", X)
print("\nTarget", y)


# splitting our features using scikit learn train-test-split method
# 70 % assigned in X_train & y_train
# 30 % assigned in X_test, y_test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.03, random_state=0, stratify=y)

print("\n\n\nX_train:\n\n", X_train)
print("\n\n\nX_test:\n\n", X_test)
