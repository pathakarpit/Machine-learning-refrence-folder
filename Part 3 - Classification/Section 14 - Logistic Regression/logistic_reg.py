import numpy as np
import pandas as pd

#for splitting the dataset
from sklearn.model_selection import train_test_split
#for feature scaling
from sklearn.preprocessing import StandardScaler
#for logistic regression
from sklearn.linear_model import LogisticRegression
#for making the confusion matrix:
from sklearn.metrics import confusion_matrix
#for checking the accuracy
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#splitting the test case 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)

#feature scaling(not necessary, only improves the result)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#making logistic regression 
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train,y_train)

#predicting a new result
print(classifier.predict( sc.transform([[30,87000]])))
print()

#predicting the test results
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
print()

#making the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
print()

#printing the accuracy
print(accuracy_score(y_test, y_pred))


