import pandas as pd

def getData():
    data = pd.read_csv("data/House_Prices.csv")
    return data


