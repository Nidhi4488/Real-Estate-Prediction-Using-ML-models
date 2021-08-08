import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
data= pd.read_csv("realestatedata.csv")
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values
y= y.reshape(len(y), 1)
# print(data.head())
# print(data.describe())
# print(data.info())
# print(data.shape())
# data.hist(bins=5, figsize=(30,20), color='yellow', rwidth= 0.95)
# plt.show()

#missing value
impute= SimpleImputer( missing_values= np.nan, strategy='mean')
impute.fit(x[:, :])
x[:,:]= impute.transform(x[:, :])





x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2, random_state=0)

#Feature scaling
sc= StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

print(x_train)
print(x_test)

# Stratified Shuffle split
split= StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
for train_index, test_index in split.split(data, data['CHAS']):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]

print(strat_test_set['CHAS'].value_counts())
print(strat_train_set['CHAS'].value_counts())
# print(95/7)
# print(376/28)

# print(len(y_train))
# print(len(y_test))

#Looking for Correlations
# Standard Correlation Coefficient
# positive means strong positive correlation, negative means strong negative correlation
correl= data.corr()
print(correl['MEDV'].sort_values(ascending=False))

#correlation Pandas plotting
#Selecting particular important features for plottig correlations
attributes= ["MEDV","RM","ZN","LSTAT"]
scatter_matrix(data[attributes], figsize=(12,12))
# plt.show()
# Plotting single graph showing correlation between two Important attributes
data.plot(kind="scatter", x="RM", y="MEDV",alpha=0.8)
# plt.show()

#Trying out new combination
data["TAXRM"]= data["TAX"]/data["RM"]
print(data["TAXRM"])
print(data.head())
correl2=data.corr()
print(print(correl2['MEDV'].sort_values(ascending=False)))
#We got new attribute which has highly strong negative correlation

data.plot(kind='scatter', x="TAXRM", y="MEDV", alpha=0.8)
# plt.show()



#model training
model= LinearRegression()
model.fit(x_train, y_train)
pred_value=model.predict(x_test)
print(pred_value[:5])
print(y_test[:5])

#Evaluating our model
mse= mean_squared_error(y_test,pred_value)
rmse= np.sqrt(mse)
print("The Rmse value of the Linear model is:"+str(rmse))


#training our datasets on decisiontree model
Dregressor= DecisionTreeRegressor(random_state=0)
Dregressor.fit(x_train, y_train)
pred_value2=Dregressor.predict(x_test)

mse2= mean_squared_error(y_test,pred_value2)
rmse2= np.sqrt(mse2)
print("The Rmse value of the Decision Tree model is:"+str(rmse2))
