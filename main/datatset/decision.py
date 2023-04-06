import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
df= pd.read_csv("C:/Users/Windows/Documents/miniproject/diabetes.csv")
df.head()
x=df.drop(['Outcome'],axis=1)
x
y=df.Outcome
y
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
model = DecisionTreeClassifier()
model = model.fit(x_train,y_train)
y_pred = model.predict(x_test)
from sklearn import metrics
print("Accuracy",metrics.accuracy_score(y_test, y_pred)*100)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)
print("Accuracy with confusion matrix:",(76+28)/154)
model.predict([[6,148,72,35,0,33.6,0.627,50]])
model = DecisionTreeClassifier(criterion="entropy", max_depth=6)
model = model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print ("accuracy:",metrics.accuracy_score(y_test, y_pred)*100)
