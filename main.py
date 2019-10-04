import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Data view configuration
desired_width=320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns',13)




##Lets start by loading our dataet to see what we are dealing with
df = pd.read_csv('wineanalysis.csv')
#print(df.head())
#We can go ahead and drop Unnamed file since there is no use in this dataset
df= df.drop(["Unnamed: 0"],axis=1)
#Lets check it out again...
print(df.head())
