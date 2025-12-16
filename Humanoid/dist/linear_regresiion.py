import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# creating a dummy dataset
np.random.seed(10)
x = np.array([[5], [10], [15], [20]])
y1 = np.array([[0.019615385], [0.08351585], [0.17745098], [0.250076336]])
y2 = np.array([[0.000138462], [0.000288184], [0.000392157], [0.000610687]])
y3 = np.array([[9.3099], [11.1444], [12.9788], [14.8133]])

# #scatterplot
# plt.scatter(x,y,s=10)
# plt.xlabel('Feed Rate')
# plt.ylabel('Average Fx')
# plt.show()

#creating a model

# creating a object
regressor1 = LinearRegression()
regressor2 = LinearRegression()
regressor3 = LinearRegression()

#training the model
regressor1.fit(x, y1)
regressor2.fit(x, y2)
regressor3.fit(x, y3)

#using the training dataset for the prediction
pred1 = regressor1.predict(x)
pred2 = regressor2.predict(x)
pred3 = regressor3.predict(x)

#model performance

# mse = mean_squared_error(y3, pred3)
# r2 = r2_score(y3, pred3)
m1 = regressor1.coef_
c1 = regressor1.intercept_
m2 = regressor2.coef_
c2 = regressor2.intercept_
m3 = regressor3.coef_
c3 = regressor3.intercept_

Y1 = m1*x + c1
Y2 = m2*x + c2
Y3 = m3*x + c3

plt.scatter(x,y1,s=20,color='Black')
plt.plot(x, Y1, color = 'Black')
plt.xlabel('Current (A)')
plt.ylabel('MRR (gm/min)')
plt.show()
plt.scatter(x,y2,s=20,color='Black')
plt.plot(x, Y2, color = 'Black')
plt.xlabel('Current (A)')
plt.ylabel('TWR (gm/min)')
plt.show()

plt.show()

#Results
#print("Mean Squared Error : ", mse)
# print("R-Squared :" , r2)
# print("Y-intercept :"  , c)
# print("Slope :" , m)