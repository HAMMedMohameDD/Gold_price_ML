# Import Necssaries Libraries

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


sns.set(rc={'figure.figsize': [9, 9]}, font_scale=1.2)

df=pd.read_csv("gld_price_data.csv")
df.head()

df.info()

df.Date = df.Date.apply(pd.to_datetime)

df.info()

# Data processing

df['Year'] = pd.DatetimeIndex(df['Date']).year # year column for extract some insights
df.head() 

df.groupby('Year').mean()

# Data visualization

plt.title('Average Gold Prices over the years')
sns.barplot(x='Year', y='GLD', data=df)

plt.title('Average Silver Prices over the years')
sns.barplot(x='Year', y='SLV', data=df)

plt.title('EUR price to USD over the years')
sns.barplot(x='Year', y='EUR/USD', data=df)

plt.title('Average USO over the years')
sns.barplot(x='Year', y='USO', data=df)

plt.title('Average SPX over the years')
sns.barplot(x='Year', y='SPX', data=df)

df2 = df.copy()
df2.drop('Year', axis=1, inplace=True) # don't need the year column 
df2.head()

sns.heatmap(df2.corr(), annot=True)

sns.distplot(df['GLD'], color = 'red')

sns.distplot(df['SLV'], color = 'red')

# Assign Feature and Target variable

x =df[['SPX','USO','SLV','EUR/USD']]
x

y = df['GLD']
y

# Splitting the data into Training and Testing data

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

scaler= StandardScaler()

scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# Modeling

models = {
    "LR": LinearRegression(),
    "KNNR" : KNeighborsRegressor(), 
    "SVR": SVR(),
    "DT": DecisionTreeRegressor(),
    "RF": RandomForestRegressor()
}

for name, model in models.items():
    print(f'Using model: {name}')
    model.fit(x_train, y_train)
    print(f'Training Score: {model.score(x_train, y_train)}')
    print(f'Test Score: {model.score(x_test, y_test)}')  
    print('-'*20)


model =  KNeighborsRegressor()

model.fit(x_train, y_train)

print(x_test)

y_pred = model.predict(x_test)
y_pred

y_test

x.columns

custom_data = np.array([2725.780029,14.405800,15.4542,1.182033])

model.predict([custom_data])

import joblib
joblib.dump(scaler, 'scaler.h5')
joblib.dump(model, 'Gold_price.model')

model2= joblib.load('Gold_price.model')

y_pred = model2.predict([[2725.780029,14.405800,15.4542,1.182033]])