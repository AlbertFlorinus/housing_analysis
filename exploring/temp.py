
import re
import pandas as pd
from matplotlib import pyplot
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

sns.set()

#load the data from my drive 
df_train = pd.read_csv('house-data/train.csv')

#delete columns with many missing data
df_train.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','LotFrontage'], axis = 1,inplace=True)

#Drop rows with missing data 
df_train.dropna(inplace=True)

from sklearn.model_selection import train_test_split
df_train = pd.get_dummies(df_train) #Getting dummies for categorical values

#Splitting test and train
X_train, X_test, y_train, y_test = train_test_split(df_train.loc[:, df_train.columns != 'SalePrice'], df_train['SalePrice'], test_size=0.25, random_state=42)
print(X_train)