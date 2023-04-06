
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
# implement a sigmoid function by hand
def sigmoid(x):
    a = []
    for item in x:
        a.append(1/(1+np.exp(-item)))
    return a
# evaluate the sigmoid at some x value
sigm = np.arange(-22, 22, 0.5)
# plot the sigmoid
plt.plot(sigm*0.2+4.57, np.array(sigmoid(sigm)), color = "red") 
# manually implemented sigmoid
plt.plot([0,10], [0.5, 0.5], linestyle = "dotted", color = "black")
plt.title("Sigmoid function")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
# split data into features/inputs and targets/outputs
feature_cols = ['pregnancies', 'insulin', 'bmi',
'age', 'glucose', 'bp', 'pedigree']
X = diabetes[feature_cols] # features
y = diabetes.outcome # target variable
# split data into training and validation datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
random_state=0)
from sklearn.linear_model import LogisticRegression
# instantiate the model
model = LogisticRegression()
# fitting the model
model.fit(X_train, y_train)
coefficents = {"Features": ["Intercept"] + feature_cols,
"Coefficients":np.concatenate((model.intercept_ ,model.coef_[0]))}
coefficents = pd.DataFrame(coefficents)
coefficents
y_pred = model.predict(X_test)
y_pred[0:5]
# metrics
print("Accuracy for test set is {}.".format(round(metrics.accuracy_score(y_test,
y_pred), 4)))
print("Precision for test set is {}.".format(round(metrics.precision_score(y_test, y_pred), 4)))
print("Recall for test set is {}.".format(round(metrics.recall_score(y_test,
y_pred), 4)))
print(metrics.classification_report(y_test, y_pred))
#confusion matrix
conf_mat = metrics.confusion_matrix(y_test, y_pred)
# plotting the confusion matrix
plt.figure(figsize=(12,6))
plt.title("Confusion Matrix")
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.ylabel("Actual Values")
plt.xlabel("Predicted Values")
plt.savefig('confusion_matrix.png')
# ROC curve
y_pred_proba = model.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr, tpr, label="auc = " + str(round(auc,2)))
plt.plot(x, y, color = "violet", label = "Ideal")
plt.plot([0,1], [0,1], color = "red", linestyle = "--", label = "random")
plt.legend(loc=4)
plt.show()
