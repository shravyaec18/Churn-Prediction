# -*- coding: utf-8 -*-
"""
Created on Sat May 29 10:43:34 2021

@author: shrut
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score,accuracy_score,roc_curve,auc
random_state = 123
import pickle


df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

df.drop(["customerID","gender"], inplace = True, axis = 1)
    
df.TotalCharges = df.TotalCharges.replace(" ",np.nan)
df.TotalCharges.fillna(0, inplace = True)
df.TotalCharges = df.TotalCharges.astype(float)
    
cols1 = ['Partner', 'Dependents', 'PaperlessBilling', 'Churn', 'PhoneService']
for col in cols1:
    df[col] = df[col].apply(lambda x: 0 if x == "No" else 1)
   

df.MultipleLines = df.MultipleLines.map({'No phone service': 0, 'No': 0, 'Yes': 1})
    
cols2 = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
for col in cols2:
    df[col] = df[col].map({'No internet service': 0, 'No': 0, 'Yes': 1})
    
df = pd.get_dummies(df, columns=['InternetService', 'Contract', 'PaymentMethod'], drop_first=True)

X = df.loc[:, df.columns != 'Churn']
y = df.loc[:, df.columns == 'Churn']
print('Shape of X: {}'.format(X.shape))
print('Shape of y: {}'.format(y.shape))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

param_grid = { 
    'n_estimators': [50, 100, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [2,4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}

rfc=RandomForestClassifier(random_state=42)

rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
rfc.fit(X_train, y_train)

rfc.best_params_

model=RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 100, max_depth=8, criterion='gini')

model.fit(X_train, y_train)

y_pred_rfc=model.predict(X_test)

print(classification_report(y_test, y_pred_rfc))




pickle.dump(model, open('model.pkl','wb'))


