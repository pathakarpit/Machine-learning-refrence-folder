import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset= pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1]
Y = dataset.iloc[:, -1]

from sklearn.imputer import SimpleImputer
imputer = SimpleImputer(missing_value=pd.nan, strategy='mean')
imputer.fit(X[:, :-1])
X[:, 1:3]= imputer.transform(X[:, 1:3])
