import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

#Display the data types of each column
fname = 'kc_house_data.csv'
df = pd.read_csv(fname)
print(df.dtypes)

#Drop the columns "id" and "date"
df.drop(['id','date'],axis=1, inplace=True)
df.describe()

#Count the number of houses with unique floor values and convert output to a dataframe
df['floors'].value_counts().to_frame()

#Use the seaborn library to produce a plot that can be used to determine whether houses with a waterfront view or without a waterfront view have more price outliers
sns.boxplot(x='waterfront',y='price',data=df)

#Use the seaborn library to determine if the feature sqft_above is negatively or positively correlated with price
sns.regplot(x='sqft_above',y='price',data=df)

from sklearn.linear_model import LinearRegression

#Fit a linear regression model to predict the price using the feature 'sqft_living' and calculate the R^2
lm = LinearRegression()
x = df[['sqft_living']]
y = df['price']

lm.fit(x,y)
rsq = lm.score(x,y)
print('The R^2 is', rsq)

#Remove missing data to fit multiple linear regression model
df.dropna(inplace =True)

missing_data = df.isnull()
for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")

#Fit a linear regression model to predict the price using the features 'floors','waterfront','lat',
#'bedrooms','sqft_basement','view','bathrooms','sqft_living15','sqft_above','grade','sqft_living' and calculate the R^2
lm1 = LinearRegression()
z = df[['floors','waterfront','lat','bedrooms','sqft_basement','view','bathrooms','sqft_living15','sqft_above','grade','sqft_living']]
lm1.fit(z,df['price'])
rsq1 = lm1.score(z,df['price'])
print('The R^2 is', rsq1)

#Create a pipeline object that scales the data, performs a polynomial transform, and fits a linear regression model.
#Fit the object using the features in the question above, then fit the model and calculate the R^2
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]
pipe = Pipeline(input)

pipe.fit(z,df['price'])
rsq2 = pipe.score(z,df['price'])
print('The R^2 is', rsq2)

#Perform a second order polynomial transform on both the training data and testing data.
#Create and fit a Ridge regression object using the training data, setting the regularisation parameter to 0.1. Calculate the R^2 utilising the testing data.
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

y_data=df['price']
x_data=df.drop('price',axis=1)

x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.15,random_state=1)
print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])

pr = PolynomialFeatures(degree=2)

x_train_pr=pr.fit_transform(x_train[['floors','waterfront','lat','bedrooms','sqft_basement','view','bathrooms','sqft_living15','sqft_above','grade','sqft_living']])
x_test_pr=pr.fit_transform(x_test[['floors','waterfront','lat','bedrooms','sqft_basement','view','bathrooms','sqft_living15','sqft_above','grade','sqft_living']])

ridgemodel= Ridge(alpha=0.1)

ridgemodel.fit(x_train_pr,y_train)
rsq3 = ridgemodel.score(x_test_pr,y_test)
print('The R^2 is', rsq3)
