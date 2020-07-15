##Data Pre-processing
#Importing appropriate libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,[13]].values

#Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])

labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_1.fit_transform(X[:,2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X =  onehotencoder.fit_transform(X).toarray()

#Dummy Variable Trap
X = X[:, 1:]


#Splitting the data into training-set and test-set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

##Creating our ANN
#importing the tools/utilities from Keras
import keras
from keras.models import Sequential
from keras.layers import Dense

#initializing our ANN
classifier = Sequential()

classifier.add(Dense(input_dim = 11, units = 6, activation="relu", kernel_initializer = "uniform"))

classifier.add(Dense(units = 6, activation="relu", kernel_initializer="uniform"))

classifier.add(Dense(units = 6, activation="relu", kernel_initializer="uniform"))

classifier.add(Dense(units = 1, activation="sigmoid", kernel_initializer="uniform"))

#Compiling the ANN
classifier.compile(optimizer = keras.optimizers.Adam(learning_rate=0.008), loss = 'binary_crossentropy', metrics =['accuracy'])
#classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


#Train the ANN
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 35, validation_data = (X_test, y_test))


#Prediction susing the test-data

y_pred =      classifier.predict(X_test)
y_pred = (y_pred>=0.5)

#Checking the accuracy prediction for the test-set
from sklearn.metrics import confusion_matrix
cm =   confusion_matrix(y_test, y_pred)




















