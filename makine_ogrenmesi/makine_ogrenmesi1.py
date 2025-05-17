import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


veriler = pd.read_csv('eksikveriler.csv')
#pd.read_csv('veriler1.csv')
print(veriler)
boy = veriler[['boy']]
print(boy)


boykilo = veriler[['boy','kilo']]
print(boykilo)

class insan:
    boy = 100
    def kosmak(self,b):
        return b+10
    
    

ali = insan()
print(ali.boy)
print(ali.kosmak(90))


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan,strategy='mean')

Yas = veriler.iloc[:,1:4].values
print(Yas)
imputer=imputer.fit(Yas[:,1:4]) 
Yas [:,1:4] = imputer.transform(Yas[:,1:4])
print(Yas)
































