import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dataframe import dataFrame  #our class to view and manipulate our data

#lets start by opening our dataset and see what columns (Features) we are dealing with...
df = dataFrame('wineanalysis.csv')
data_frame = df.df
#df.see_info() #We see an col named Unnamed... lets get rid of it...
df.drop('Unnamed: 0')

#We need to convert our class lable (type) into numerical values... this is part of preprocessing our data
#There is no convienient function to do this, so we will be using mapping
type_map = {
    'red' :1,
    'white':0
}

#replaced Red types for 1 and white as 0
df.map_features(type_map, 'type')
df.see_info() #All of our data is now numerical







