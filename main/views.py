from django.shortcuts import render, redirect
from django.http import HttpResponse

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# load the diabetes dataset
df = pd.read_csv("C:/Users/Windows/Desktop/esdp/diabetes.csv")

# prepare the input features (drop the Outcome column)
X = df.drop(['Outcome'], axis=1)

# prepare the target variable
y = df.Outcome

# create the decision tree model
model = DecisionTreeClassifier(criterion="entropy", max_depth=6)
model = model.fit(X, y)


def home(request):
    
    if request.method == "GET":
        return render(request, "index.html")
    else:
        age = int(request.POST.get('age'))
        bmi = float(request.POST.get('bmi'))
        glucose = float(request.POST.get('glucose'))
        bp = float(request.POST.get('bp'))
        skin = float(request.POST.get('skin'))
        insulin = float(request.POST.get('insulin'))
        new_data = np.array([[glucose, bp, skin, insulin, bmi, age]])
        
        
        print(new_data)
        prediction = model.predict(new_data)
        print("########################")
        print(prediction)
        if  any(prediction)==1:
            msg=1
        else:
            msg=0
        context={}
        prediction=None
        context['msg']=msg
        return render(request,"result.html",context)

def about(request):
    return render(request, 'about.html')

def contact(request):
    return render(request, 'contact.html')
# Create your views here.
