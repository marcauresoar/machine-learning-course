# Data Preprocessing Template

# Importing the libraries

## contains mathematical tools
import numpy as np
## used to plot charts
import matplotlib.pyplot as plt 
## best data to import and manage datasets
import pandas as pd 


# Importing the dataset
dataset = pd.read_csv('./Data.csv')
# print(dataset)

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values
# print(x)

# Taking care of missing data
# sklearn = lib that contains a set of tools to create machine learning models
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])
# print(x)

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()
# print(x)

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
# print(y)


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

# print(x_train)
# print(x_test)


