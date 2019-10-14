# Red-Or-White-Wine
A classification data set that classifies descriptions on what attributes make red or white wine
The Plan:
once selected the data, we will figure out if we need to pre-process it. 

We will split the data into 70% training,  15% Development,  15% Tests


Then we will think about what the most appropriate evaluation metric is for the dataset (Accuracy VS F1).


The Specifics:

1) will use Logistic Regression or SVM implementations in scikit-learn to train our classifier(Default parameters) 
and will evaluate our classifier on the development set

2) next we will tweak our parameters and see if we can improve our model's performance on the development set

3) We will then implement our own K-nearest neighbors classifier (KNN) and tune ethe k value to achieve the best
possible performance on the development set

4) finally, we will compare our best model in step 2 to our best KNN model by evaluating them on our test set


Based of Python 3 

To run, type in terminal --> python3 main.py

NOTE: KNN model will take a minute to fully "train"

3rd party installments:

Numpy & Pandas
