import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

dataset = pd.read_csv('saber_pro.csv',delimiter=',')

dataset['ESTU_GENERO'] = dataset['ESTU_GENERO'].map({'F':1,'M':0})

x = dataset.iloc[:, [60,63,66,69,72]].values 
y = dataset.iloc[:, 2].values 

xtrain, xtest, ytrain, ytest = train_test_split( x, y, test_size = 0.40) 

sc_x = StandardScaler() 
xtrain = sc_x.fit_transform(xtrain)  
xtest = sc_x.transform(xtest) 


classifier = LogisticRegression(solver='lbfgs') 
classifier.fit(xtrain, ytrain) 

clf=classifier.fit(xtrain, ytrain) 

y_pred = classifier.predict(xtest) 

print(y_pred)
cm = confusion_matrix(ytest, y_pred) 

print ("Confusion Matrix : \n", cm) 

print ("Accuracy : ", accuracy_score(ytest, y_pred)) 

#Grafica
points_x=[x/10. for x in range(-1009,+1000)]

line_bias = clf.intercept_
line_w = clf.coef_.T
points_y=[(line_w[0]*x+line_bias)/(-1*line_w[1]) for x in points_x]
plt.plot(points_x, points_y)
plt.scatter(x[:,0], x[:,1],c=y)
plt.show()
