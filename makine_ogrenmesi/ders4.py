# Çoklu doğrusal regresyon Örnek 1

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ders2 import x_yeni

veriseti = pd.read_excel('veriler/Pv.xlsx')
X = veriseti.iloc[:,:-1].values
y = veriseti.iloc[:,-1].values

uzunluk = len(X)
X = np.append(np.ones((uzunluk,1)).astype(float), values=X, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.linear_model import LinearRegression
model_Regresyon = LinearRegression()
model_Regresyon.fit(X_train, y_train)

print(model_Regresyon.coef_, model_Regresyon.intercept_)
#%%
import statsmodels.api as sm
model_Regresyon_OLS = sm.OLS(y_train, X_train).fit()
print(model_Regresyon_OLS.summary())
#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

np.random.seed(0)

# Veri oluşturma
X = np.array([3,6,12,16,22,28,33,40,47,51,55,60])
y = np.array([20,28,50,64,67,57,59,59.5,68,74,80,90])

# Veri görselleştirme
plt.figure(1)
plt.scatter(X, y, color='red', marker='o')
plt.show()
# Polinom özellikler oluşturma
model_Polinom_Regresyon = PolynomialFeatures(degree=3)
X_polinom = model_Polinom_Regresyon.fit_transform(X.reshape(-1,1)) #işlemi, tek boyutlu (1D) bir diziyi iki boyutlu (2D) hale getirmek için yapılıyor.

# Linear regresyon ile polinom model oluşturma
model_Regresyon = LinearRegression()
model_Regresyon.fit(X_polinom, y)

print(model_Regresyon.coef_)
print(model_Regresyon.intercept_)

#%%
# OLS ile model istatistiği
import statsmodels.api as sm
model_Regresyon_OLS = sm.OLS(y, X_polinom).fit()
print(model_Regresyon_OLS.summary())
#%%
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
X = np.array([3,6,12,16,22,28,33,40,47,51,55,60])
y = np.array([20,28,50,64,67,57,59,59.5,68,74,80,90])
# Veri görselleştirme
plt.figure(1)
plt.scatter(X, y, color='red', marker='o')
plt.show()
model_Polinom_Regresyon = PolynomialFeatures(degree=3)
X_polinom = model_Polinom_Regresyon.fit_transform(X.reshape(-1,1))
model_Regresyon = LinearRegression()
model_Regresyon.fit(X_polinom, y)
# Tahmin için grid oluşturma ve çizim
Xgrid = np.arange(min(X), max(X), 0.1)
Xgrid = Xgrid.reshape(-1, 1)
ypred = model_Polinom_Regresyon.fit_transform(Xgrid)
ypred = model_Regresyon.predict(ypred)
plt.plot(Xgrid, ypred, color='blue')
plt.show()
#%%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Veriyi oku
veriseti = pd.read_excel('veriler/Pv.xlsx')
X = veriseti.iloc[:,:-1].values
y = veriseti.iloc[:,-1].values

# Bias (sabit terim) ekle
uzunluk = len(X)
X = np.append(np.ones((uzunluk,1)).astype(float), values=X, axis=1)

# Veriyi böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Modeli eğit
model_Regresyon = LinearRegression()
model_Regresyon.fit(X_train, y_train)

# Tahmin yap
ytahmin = model_Regresyon.predict(X_train)

# Model başarısını ölç
print("R^2 Skoru:", r2_score(y_train, ytahmin))


# Test verileri ile model performans ölçümü
y_test_tahmin = model_Regresyon.predict(X_test)
print(r2_score(y_test, y_test_tahmin))

#%%
import numpy as np
# Yeni verilerle tahmin
x = [1,6,5,4.5]
X_yeni = np.array([1,6,5,4.5])
X_yeni = X_yeni.reshape(-1,1)
X_yeni = model_Polinom_Regresyon.fit_transform(X_yeni)
model_Polinom_Regresyon.fit(X_yeni, y_test)

print("tahminler",model_Regresyon.predict(X_yeni))
