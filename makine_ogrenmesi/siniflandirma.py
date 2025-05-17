# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 15:13:28 2024

@author: samet
"""

#1.Kütüphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



veriler = pd.read_csv('veriler.csv')
print(veriler)

x = veriler.iloc[:,1:4].values #bağımzın değişkenler
y = veriler.iloc[:,4:].values   #bağımlı değişkenler



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=0)



#öznitelik Ölçekleme
 
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


#Logistic regresyon
print('Logistic regresyon')

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)
print(y_pred)
print(y_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)



#KNN
print('KNN')

from sklearn.neighbors import KNeighborsClassifier

knn =KNeighborsClassifier(n_neighbors=1,metric='minkowski')
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)

#destek vektör 
print('destek vektör')

from sklearn.svm import SVC

svc = SVC(kernel='rbf')
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)

# naive bayes
print('naive bayes')

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)

y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)

#karar ağaçları 
print('karar ağaçları')

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='entropy')
dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)


#rassal ağaçlar
print('rassal ağaçlar')

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=9,criterion='entropy')
rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)



