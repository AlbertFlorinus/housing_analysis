import pandas as pd

df = pd.read_csv("house-data/train.csv")
#Dropping alley col, too much missing values
df = df.drop(columns=["Alley"])

print(df.dtypes)
#print(df.keys())