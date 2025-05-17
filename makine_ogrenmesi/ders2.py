import numpy as np
from sklearn.preprocessing import MinMaxScaler
x=np.array([22,87,20,91,48,61,76,51,29,18])
scaler = MinMaxScaler()
y=scaler.fit_transform(x.reshape(-1,1))

#%%
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets

iris=datasets.load_iris()
x=iris.data[:,0]
scaler = MinMaxScaler()
y=scaler.fit_transform(x.reshape(-1,1))

#%%
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

iris=datasets.load_iris()
x=iris.data[:,0]
scaler = StandardScaler()
y=scaler.fit_transform(x.reshape(-1,1))

#%%
import pandas as pd
from sklearn.preprocessing import LabelEncoder

veriseti = pd.read_excel("veriler/verionisleme_2.xlsx")
#kategorik veriye dönüştürür ve x ve y'yi ayırma
x=veriseti.iloc[:,:-1].values
y=veriseti.iloc[:,-1].values

LabelEncoder_x=LabelEncoder()
x[:,0]=LabelEncoder_x.fit_transform(x[:,0])
print(x)

#%%
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

veriseti = pd.read_excel("veriler/verionisleme_2.xlsx")
# x ve y'yi ayırma kategorik veriye dönüştürür 1 0 şekilinde yapılır!
x=veriseti.iloc[:,:-1].values
y=veriseti.iloc[:,-1].values

columntransformer_x = ColumnTransformer(([("encoder",OneHotEncoder(),[0])]),remainder='passthrough')
x=columntransformer_x.fit_transform(x)
print(x)

#%%
#Basit Doğrusal Regresyon
#baş çevresine göre beyin ağırlığı. csv dosyası ile ilgili soru gelebilir
#y1=bo+b1x1
#y2=b0+b2x2 bu çok değişkenli
#y=bo+b1x1+b2x2 bu da çoklu regresyon
import numpy as np
import pandas as pd

veriseti = pd.read_csv("veriler/DogrusalRegresyon.csv")

x=veriseti['Bas_cevresi(cm^3)'].values
y=veriseti['Beyin_agirligi(gr)'].values

uzunluk=len(x)
x=x.reshape((uzunluk,1))
x=np.append(arr=np.ones((uzunluk,1)).astype(int),values=x,axis=1)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
from sklearn.linear_model import LinearRegression

model_regresyon=LinearRegression()
model_regresyon.fit(x_train,y_train)

print(model_regresyon.coef_)
print(model_regresyon.intercept_)

#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

veriseti = pd.read_csv("veriler/DogrusalRegresyon.csv")

x=veriseti['Bas_cevresi(cm^3)'].values
y=veriseti['Beyin_agirligi(gr)'].values

uzunluk=len(x)
x=x.reshape((uzunluk,1))
x=np.append(arr=np.ones((uzunluk,1)).astype(int),values=x,axis=1)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
from sklearn.linear_model import LinearRegression

model_regresyon=LinearRegression()
model_regresyon.fit(x_train,y_train)
# y_test -> y_true  ; y_train -> y_pred (tahmin)
# modelin doğru denklemi
plt.figure(1)
plt.scatter(x_train[:,1],y_train,color="red",marker="o") #Eğitim veri setindeki gerçek değerleri (kırmızı daireler) grafiğe yerleştirir.
pred_x_train=(model_regresyon.intercept_)+(model_regresyon.coef_[1])*x_train[:,1]
plt.plot(x_train[:,1],pred_x_train,color="blue")
plt.show()

plt.figure(1)
plt.scatter(x_test[:,1],y_test,color="red",marker="o")
pred_x_test=(model_regresyon.intercept_)+(model_regresyon.coef_[1])*x_test[:,1]
# Regresyon doğrusunun Y eksenini kestiği nokta (sabit terim).
# İkinci özelliğin (indeks 1) katsayısı.
#Burada y = b₀ + b₁x₁ formülünü uyguluyorsunuz.
plt.plot(x_test[:,1],pred_x_test,color="blue")
#Hesaplanan tahmin değerlerini mavi bir çizgi olarak grafiğe ekler.
plt.show()

#%%

from sklearn.metrics import mean_absolute_error,r2_score

print(mean_absolute_error(y_test,pred_x_test))
print(r2_score(y_test,pred_x_test))

pred_x_train = model_regresyon.predict(x_train)
print(r2_score(y_train,pred_x_train))

#%%

# Anova tablosu oluşturma
#No. Observations (Gözlem Sayısı): Veri kümesindeki toplam gözlem (örneklem) sayısı.
#0.05’ten küçükse, model genel olarak anlamlıdır.
#[0.025, 0.975] Güven Aralığı: Katsayının %95 güven aralığını gösterir.

# Test	                      Açıklama	                                                   İdeal Değer / Aralık
# Omnibus p-değeri	      Hata terimlerinin normal dağılıp dağılmadığını test eder.	       > 0.05
# Skewness (Çarpıklık)	  Dağılımın simetrik olup olmadığını gösterir.	                   -0.5 ile +0.5 (maksimum -1 ile +1)
# Kurtosis (Basıklık)	  Dağılımın sivri veya basık olup olmadığını gösterir.	           2 ile 4
# Durbin-Watson (DW)	   Otokorelasyonu kontrol eder.	                                   1.5 ile 2.5
# Jarque-Bera p-değeri	  Hata terimlerinin normal dağılıma uyup uymadığını test eder.	   > 0.05
import statsmodels.api as sm

model_regresyon_OLS = sm.OLS(y_train,x_train).fit()
print(model_regresyon_OLS.summary())



#%%
import numpy as np
# x = [3200,4500,3879] için ypred = ?

x_yeni =np.array([[1,3200],[1,4500],[1,3879]])
y_tahmin = model_regresyon.predict(x_yeni)
print(y_tahmin)

y_tahmin1 = 354.8403+0.2553*3200
print(y_tahmin1)

#%%

import numpy as np

x = np.array([8,10,12,14,16])
y= np.array([20,24,25,26,30])

uzunluk=len(x)
x = x.reshape((uzunluk,1))
x = np.append(arr=np.ones((uzunluk,1)).astype(int),values=x,axis=1)

from sklearn.linear_model import LinearRegression
model_regresyon = LinearRegression()
model_regresyon.fit(x,y)

print(model_regresyon.coef_) # Bağımsız değişkenlerin katsayılarını verir. (Eğim)
print(model_regresyon.intercept_) #b0

from sklearn.metrics import r2_score
y_tahmin = model_regresyon.predict(x)
R2 = r2_score(y,y_tahmin)
print(R2)
