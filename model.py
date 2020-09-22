# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:59:01 2020

@author: Tushar
"""

import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

df = pd.read_csv("insurance.csv")

df.drop(['region','sex','smoker'], axis='columns', inplace=True)

x = df.drop('charges',axis = 1)
y = df['charges']


#x_train,x_test,y_train,y_test = train_test_split(x,y) #,test_size = 0.2)

reg = LinearRegression()
reg.fit(x,y)

print(reg.score(x,y))
#print(reg.score(x_test,y_test))

pickle.dump(reg, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[19, 27.9, 3]]))