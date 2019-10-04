import pandas as pd

# Data view configuration
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 20)


# Class just to make things slightly easier to read and handle organize our code...
class DataFrame():
    def __init__(self, csv):
        self.df = pd.read_csv(csv)

    # any methods we need for our data will be here
    def see_info(self):
        print(self.df.info())

    def head(self, peek=5):
        print(self.df.head(peek))

    def tail(self, peek=5):
        print(self.df.tail(peek))

    def drop(self, col):
        self.df = self.df.drop([col], axis=1)
        return self.df

    def map_features(self, map, col):
        self.df[col] = self.df[col].map(map)

    def select_data_points(self, feat, targ):
        X = self.df.iloc[:, :feat].values
        y = self.df.iloc[:, targ].values
        return X,y
