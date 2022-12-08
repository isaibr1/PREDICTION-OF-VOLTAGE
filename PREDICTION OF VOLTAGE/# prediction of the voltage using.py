# predict the voltage of electricity using linear regretion technique
# NAME: ISA IBRAHIM
# COURSE : TRUSTWORTHY MACHINE LEARNING
# PROJECT CODE

# import library

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import numpy as np
import pandas as pd


df = pd.read_csv(r'C:\Users\User\Desktop\voltage.csv')


#Breaki date column into multiple columns
df["Data"]=pd.to_datetime(df["Data"])
df["Data1"]=df["Data"].dt.strftime("%d.%m.%Y %H:%M:%S")
df["Year"]=pd.DatetimeIndex(df["Data1"]).year
df["Month"]=pd.DatetimeIndex(df["Data1"]).month
df["Day"]=pd.DatetimeIndex(df["Data1"]).day
df["Data1"]=df["Data"].dt.strftime("%H:%M:%S")
df["HTime"]=pd.DatetimeIndex(df["Data"]).hour
df["MTime"]=pd.DatetimeIndex(df["Data"]).minute



df=df.drop(["Data", "Data1"],axis=1) #drop initial column

# looping the output data date and voltage

x = df.iloc[1 : , 1: 6]#1 : , 1 : 2
y = df.iloc[1 : , 0 : 1]

# Print the date and voltage
print(x)
print(y)

# The training algorithsm (linearregression, lasso, and ridge)
Linear_model = LinearRegression()
Lasso_model = Lasso(alpha = 1)
Ridge_model = Ridge(alpha = 1)

# create and fit the data to the model  
Linear_model.fit(x,y)
Lasso_model.fit(x,y)
Ridge_model.fit(x,y)


# Test data
x_new = [[2009, 10, 4, 11, 23]]

# Make prediction
print("Linear Regression = ", Linear_model.predict(x_new))
print("Lasso Reguarization:", Lasso_model.predict(x_new))
print("Ridge Reguarization:", Ridge_model.predict(x_new))
