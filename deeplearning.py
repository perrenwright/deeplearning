#Deep Learning
import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
import scipy
import sklearn

#Reading in the data
print("Breast Cancer Wisconsin (Diagnostic) Data Set")

#datasets from the website 
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"


#32 attributes
attributes = ["ID", "Diagnosis", "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness", "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension", 
"radius se", "texture se", "perimeter se", "area se", "smoothness se", "compactness se", "concavity se", "concave points se", "symmetry se", "fractal dimension se", 
"radius worst", "texture worst", "perimeter worst", "area worst", "smoothness worst", "compactness worst", "concavity worst", "concave points worst", "symmetry worst", "fractal dimesion worst"]

df = pd.read_csv(url, names = attributes)
print(df)

array = df.values 

X = array[:569, 2:32]
y = array[:569, 1]

#transformed to fit the mean value of a column
imputer = Imputer()
X_transf = imputer.fit_tranform(X)

#Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X_transf, y, test_size = 0.3)
