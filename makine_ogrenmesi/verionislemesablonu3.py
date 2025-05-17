
#kütüphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#veri yukleme
veriler = pd.read_csv('maaslar.csv')


#data frame dilimle(slice)
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]

#Numpy dizi array donusumu
X = x.values
Y = y.values


# linear regression
#doğrusal model oluşturma
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X,Y)

from sklearn.metrics import r2_score
print('linear regression R2 Degeri')
print(r2_score(Y, lin_reg.predict(X)))


#doğrusal olmayan model oluşturma
#polynomial regression
#2.dereceden
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)

#4.dereceden denenme kısmı
poly_reg3 = PolynomialFeatures(degree=4)
x_poly3 = poly_reg3.fit_transform(X)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly3, y)


#görseleştirme
plt.scatter(X, Y,color='red')
plt.plot(x,lin_reg.predict(X),color='blue')
plt.show()

plt.scatter(X, Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()

plt.scatter(X, Y,color = 'red')
plt.plot(X,lin_reg3.predict(poly_reg3.fit_transform(X)), color = 'blue')
plt.show()



#tahmimler 

print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))
print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))


from sklearn.metrics import r2_score
print('polynomial regression R2 Degeri')
print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X))))


 #verilerin ölçeklenmesi
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_olcekli  = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)

#destek vetörü
from sklearn.svm import SVR

svr_reg = SVR(kernel = 'rbf')
svr_reg.fit(x_olcekli,y_olcekli)



plt.scatter(x_olcekli,y_olcekli,color = 'red')
plt.plot(x_olcekli,svr_reg.predict(x_olcekli), color = 'blue')
plt.show()

print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6.6]]))

from sklearn.metrics import r2_score
print('destek vetörü R2 Degeri')
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)))

#karar ağaçları regresyon
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)


plt.scatter(X, Y, color = 'red')
plt.plot(X,r_dt.predict(X), color = 'blue')
plt.show()

from sklearn.metrics import r2_score
print('karar ağaçları R2 Degeri')
print(r2_score(Y, r_dt.predict(X)))

print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))


#rassal orman regrosyon
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)
rf_reg.fit(X, Y.ravel())

print(rf_reg.predict([[6.6]]))

plt.scatter(X, Y,color = 'red')
plt.plot(X,rf_reg.predict(X),color='blue')
plt.show()

from sklearn.metrics import r2_score
print('Random Forest R2 Degeri')
print(r2_score(Y, rf_reg.predict(X)))


#ozet R2 değerleri
print('--------------------------------')
print('linear regression R2 Degeri')
print(r2_score(Y, lin_reg.predict(X)))


print('polynomial regression R2 Degeri')
print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X))))


print('destek vetörü R2 Degeri')
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)))

print('karar ağaçları R2 Degeri')
print(r2_score(Y, r_dt.predict(X)))


print('Random Forest R2 Degeri')
print(r2_score(Y, rf_reg.predict(X)))

