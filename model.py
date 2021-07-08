#Importing Libraries
import pandas as pd
import numpy as np
from sklearn import linear_model

df=pd.read_csv('car_data.csv')


inputs=df.drop(['Car_Name','Seller_Type'],axis='columns')
target=df.Selling_Price

from sklearn.preprocessing import LabelEncoder
Numerics=LabelEncoder()

inputs['Fuel_Type_n']=Numerics.fit_transform(inputs['Fuel_Type'])
inputs['Transmission_n']=Numerics.fit_transform(inputs['Transmission'])

inputs_n=inputs.drop(['Fuel_Type','Transmission','Selling_Price'],axis='columns')
model=linear_model.LinearRegression()
model.fit(inputs_n,target)

#Prediction
pred=model.predict([[2014,5,27000,0,2,1]])
print(round(pred[0],2))