# POLİNOM REGRESYONU
# örnek

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(6)

def fonksiyon(x):
    return 10*np.sin(x/2)+x/7+6

x = np.random.rand(100)*10
y = fonksiyon(x)+2*np.random.randn(*x.shape)

uzunluk = len(x)
x = x.reshape(uzunluk,1)

plt.figure(1)
plt.scatter(x,y,color='red')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn.preprocessing import PolynomialFeatures
model_polinom_regresyon = PolynomialFeatures(degree=3)
#[1,x,x^2,x^3] 1 sabit
x_polinom = model_polinom_regresyon.fit_transform(x_train)

from sklearn.linear_model import LinearRegression
model_regresyon=LinearRegression()
#Lineer regresyonu kullanıyoruz ama önceden
# x’i polinomal hale getirdiğimiz için artık polinom regresyon oluyor 🚀.
model_regresyon.fit(x_polinom,y_train)

print(model_regresyon.coef_) #[ 0.         10.63970336 -2.43102426  0.13109035]
#yani 10.63x-2.4x^2+0.131x^3
print(model_regresyon.intercept_) #sabit