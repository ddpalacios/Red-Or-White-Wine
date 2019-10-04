import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Data view configuration
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 20)
#Class just to make things slightly easier to read and handle organize our code...
class dataFrame():
    def __init__(self, csv):
        self.df = pd.read_csv(csv)

    #any methods we need for our data will be here
    def see_info(self):
        print(self.df.info())

    def peek(self, peek):
        print(self.df.head(peek))

    def drop(self, col):
        self.df = self.df.drop([col], axis=1)
        return self.df


#Now that we created our class, lets start by opening our dataset and see what columns (Features) we are dealing with
df = dataFrame('wineanalysis.csv')
#df.see_info() #We see an col named Unnamed... lets get rid of it.
df.drop('Unnamed: 0')
#lets view again...
#df.see_info()  #Great... lets get rid od this methods here and we can start handeling our selected features
