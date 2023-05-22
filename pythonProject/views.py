from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings('ignore')

def home(request):
    return render(request, 'home.html')

def predict(request):
    return render(request, 'predict.html')

def result(request):
    #getting the data
    data_frame = pd.read_csv(r"C:\Users\Lina Ben Salem\Desktop\diabetes project\diabetes.csv")
    #splitting the data
    feature_col_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                         'DiabetesPedigreeFunction', 'Age']
    predicted_class_names = ['Outcome']

    X = data_frame[feature_col_names].values  # predictor feature columns (8 X m)
    y = data_frame[predicted_class_names].values  # predicted class (1=true, 0=false) column (1 X m)
    split_test_size = 0.30

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_test_size, random_state=42)
    # test_size = 0.3 is 30%, 42 is the answer to everything
    # create Gaussian Naive Bayes model object and train it with the data
    nb_model = GaussianNB()

    nb_model.fit(X_train, y_train.ravel())

    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])

    pred = nb_model.predict([[val1,val2,val3,val4,val5,val6,val7,val8]])
    result1 = ""
    if pred == [1]:
        result1 = "Positive"
    else:
        result1 = "Negative"

    return render(request, 'predict.html',{"result2":result1})
