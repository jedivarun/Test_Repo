import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import streamlit as st

#Reading file

df_Salary = pd.read_csv('salary_data.csv')

#Y and X
X = np.array(df_Salary['YearsExperience']).reshape(-1, 1)
y = np.array(df_Salary['Salary']).reshape(-1, 1)

regr_salary = LinearRegression()
  
regr_salary.fit(X, y)

salary_pickle = open('salary_predict.pickle','wb')

pickle.dump(regr_salary,salary_pickle)

salary_pickle.close()

#pickle_a=open("salary_predict.pickle","rb")
#regressor=pickle.load(pickle_a) # our model

