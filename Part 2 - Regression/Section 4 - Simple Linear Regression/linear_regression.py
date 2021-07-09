import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#to split into training and test sets
from sklearn.model_selection import train_test_split

#to train the simple linear regression model on the training set
from sklearn.linear_model import LinearRegression


#importing the datasets
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values

#splitting dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

#training the simple linear regression model on the training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting the test results
y_pred = regressor.predict(X_test)

#visualize the training set results
plt.scatter(X_train,y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('salary vs experience(training set)')
plt.xlabel('years of expirence')
plt.ylabel('salary')
plt.show()

#visualising the test set results
plt.scatter(X_test,y_test, color= 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title("salary vs experience(test set)")
plt.xlabel('years of expirence')
plt.ylabel('salary')
plt.show()

#Making a single prediction (for example the salary of an employee with 12 years of experience)
print(regressor.predict([[12]]))

#Getting the final linear regression equation with the values of the coefficients
k = (regressor.coef_)
c = (regressor.intercept_)
print("Therefore, the equation of our simple linear regression model is:")
print("salary = "+str(k[0])+' x years + '+str(c))