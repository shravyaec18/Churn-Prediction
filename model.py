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
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score,accuracy_score,roc_curve,auc
random_state = 123
import pickle


df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

df.drop(["customerID"], inplace = True, axis = 1)
    
df.TotalCharges = df.TotalCharges.replace(" ",np.nan)
df.TotalCharges.fillna(0, inplace = True)
df.TotalCharges = df.TotalCharges.astype(float)
    
cols1 = ['Partner', 'Dependents', 'PaperlessBilling', 'Churn', 'PhoneService']
for col in cols1:
    df[col] = df[col].apply(lambda x: 0 if x == "No" else 1)
   
df.gender = df.gender.apply(lambda x: 0 if x == "Male" else 1)
df.MultipleLines = df.MultipleLines.map({'No phone service': 0, 'No': 0, 'Yes': 1})
    
cols2 = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
for col in cols2:
    df[col] = df[col].map({'No internet service': 0, 'No': 0, 'Yes': 1})
    
df = pd.get_dummies(df, columns=['InternetService', 'Contract', 'PaymentMethod'], drop_first=True)

from sklearn.ensemble import IsolationForest

model=IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.1),max_features=1.0)
model.fit(df[['tenure','TotalCharges','MonthlyCharges']])

df['scores']=model.decision_function(df[['tenure','TotalCharges','MonthlyCharges']])
df['anomaly_Value']=model.predict(df[['tenure','TotalCharges','MonthlyCharges']])

df['anomaly_Value'][df['anomaly_Value'] ==-1].count()
df= df[df['anomaly_Value'] == 1]

sc = StandardScaler()
df['MonthlyCharges'] = sc.fit_transform(df[['MonthlyCharges']].values)
df['TotalCharges'] = sc.fit_transform(df[['TotalCharges']].values)
df['tenure'] = sc.fit_transform(df[['tenure']].values)
df.head()

X = df.drop(['Churn'],axis=1)
y = df['Churn']
y.value_counts()

estimator = LogisticRegression(random_state=random_state)
rfecv = RFECV(estimator=estimator, cv=StratifiedKFold(10, random_state=random_state, shuffle=True), scoring="accuracy")
rfecv.fit(X, y)



print("The optimal number of features: {}".format(rfecv.n_features_))

X_rfe = X.iloc[:, rfecv.support_]

print("\"X\" dimension: {}".format(X.shape))
print("\"X\" column list:", X.columns.tolist())
print("\"X_rfe\" dimension: {}".format(X_rfe.shape))
print("\"X_rfe\" column list:", X_rfe.columns.tolist())

X_rfe_train, X_rfe_test, y_train, y_test = train_test_split(X_rfe,y,train_size=0.75,stratify=y,random_state=random_state)
print("Train size: {}".format(len(y_train)))
print("Test size: {}".format(len(y_test)))

lr = LogisticRegression()
model = lr.fit(X_rfe_train,y_train)
y_pred = model.predict(X_rfe_test)

print(pd.DataFrame(confusion_matrix(y_test,y_pred)))

print(classification_report(y_test,y_pred))

import pickle
pickle.dump(model, open('model.pkl','wb'))


