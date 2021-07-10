#importing the data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)

#data preprocessing
transactions = []
for i in range(7501):
    transactions.append([str(dataset.values[i,j]) for j in range(20)])


#training the apriori model
from apyori import apriori
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length=2)

#visualising the result
results = list(rules)

#putting the results in an organised manner 
def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts = [result[2][0][3] for result in results]
    return list(zip(lhs,rhs,supports,confidences,lifts))
resultinDataFrame = pd.DataFrame(inspect(results), columns = ['left hand side', 'right hand side', 'support', 'confidence', 'lift'])
print(resultinDataFrame)
print('///')

#displyaing the results sorted
print(resultinDataFrame.nlargest(n = 10, columns='lift'))
