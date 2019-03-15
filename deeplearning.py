#Deep Learning
import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
import scipy
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.metrics import f1_score
from sklearn.preprocessing import Imputer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
#Reading in the data
print("Breast Cancer Wisconsin (Diagnostic) Data Set")

#datasets from the website 
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"


#32 attributes
attributes = ["ID", "Diagnosis", "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness", "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension", 
"radius se", "texture se", "perimeter se", "area se", "smoothness se", "compactness se", "concavity se", "concave points se", "symmetry se", "fractal dimension se", 
"radius worst", "texture worst", "perimeter worst", "area worst", "smoothness worst", "compactness worst", "concavity worst", "concave points worst", "symmetry worst", "fractal dimesion worst"]

df = pd.read_csv(url, names = attributes)
df.dropna(0, how ='any')
array = df.values 

X = array[:569, 2:32]
Y_char = np.array(array[:569, 1])
ychar, y = np.unique(Y_char, return_inverse= True)

#transformed to fit the mean value of a column
imputer = Imputer()
X_transf = imputer.fit_transform(X)

#Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X_transf, y, test_size = 0.3)

#Deep Learning Model Fit
clf = MLPClassifier(solver='lbfgs', alpha = 1e-5, hidden_layer_sizes=(3, 1), random_state=1)
clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
score = clf.score(X_test, y_test)
report = classification_report(y_test, y_pred)
print(report)
print("The score is: " + str(score))
