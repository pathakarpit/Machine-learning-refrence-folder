import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#for encoding data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
#splitting the data into test set and training set
from sklearn.model_selection import train_test_split
#for training the multiple linear regression
from sklearn.linear_model import LinearRegression

#importing data
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#encoding the required (in this case cities)
ct =ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough')
X = np.array(ct.fit_transform(X))

#splitting dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#training the multiple linear regression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting the test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision = 2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

#Making a single prediction 
#(for example the profit of a startup with R&D Spend = 160000, Administration Spend = 130000, Marketing Spend = 300000 and State = 'California')
print(regressor.predict([[1, 0, 0, 160000, 130000, 300000]]))

#Getting the final linear regression equation with the values of the coefficients
print(regressor.coef_)
print(regressor.intercept_)
print('Profit = 86.6 × DummyState1 − 873 × DummyState2 + 786 × DummyState3 + 0.773 × R&DSpend + 0.0329 × Administration + 0.0366 × MarketingSpend + 42467.53')