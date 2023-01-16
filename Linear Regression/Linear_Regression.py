import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Importing Dataset
df = pd.read_csv('dhaka homeprices.csv')
# In this Dataset, Area=Independent Variable, Price=Dependent Variable
# Area = Feature and Price = Label

# print(df)
# print(df.isnull().sum())  [Checking Null Values]

x = df[['area']] # Feature always 2 dimensional, thats why 2 brackets
y = df['price'] # Label always 1 dimensional, thats why 1 bracket

# print(x)

plt.scatter(df['area'], df['price'], color='red', marker='o')
plt.xlabel('Area in Sqare feet')
plt.ylabel('Price in Taka')
plt.title('Area vs Price')
#plt.show()

# Data will be split Randomly into 2 parts, 70% for training and 30% for testing
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=1)

# print(xtrain)

# Object created and fit completed
reg = LinearRegression()
reg.fit(xtrain, ytrain)

# print(reg.predict(xtest)) # Predicting Price for xtest (area)
# print(reg.predict([[3300]])) # Predicting Price for 3300 sqft area


# Plotting the Best fit Regression Line
plt.plot(df.area, reg.predict(df[['area']]), color='blue')
plt.show()

