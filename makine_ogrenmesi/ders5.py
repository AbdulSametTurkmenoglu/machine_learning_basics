import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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
print(model_Regresyon.coef_,model_Regresyon.intercept_)
#%%
# Statsmodels OLS
import statsmodels.api as sm
model_Regresyon_OLS = sm.OLS(y_train, X_train).fit()
print(model_Regresyon_OLS.summary())
#%%
# Test data
x_yeni = X_train[:, [0, 1, 2,3,4, 5, 6, 7]]
model_Regresyon_OLS=sm.OLS(y_train, x_yeni).fit()
print(model_Regresyon_OLS.summary())

#%%
# Creating training data
x_opt = X_train[:,[0, 1, 2, 5, 6, 7]]
model_Regresyon_OLS = sm.OLS(y_train, x_opt).fit()
print(model_Regresyon_OLS.summary())
#%%
# Creating prediction metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
x_test_opt = X_test[:, [0, 1, 2, 5, 6, 7]]
model_Regresyon.fit(x_opt, y_train)
y_pred = model_Regresyon.predict(x_test_opt)
# Print evaluation metrics
print("MAE=%0.2f" % mean_absolute_error(y_test, y_pred))
print("MSE=%0.2f" % mean_squared_error(y_test, y_pred))
print("MedAE=%0.2f" % median_absolute_error(y_test, y_pred))
print("R2=%0.2f" % r2_score(y_test, y_pred))

#%%
#LOGİSTİK REGRESYON

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Veri yükleme
veri = pd.read_excel('veriler/Immunotherapy1.xlsx')

# Özellikler ve hedef değişken
X = veri.iloc[:, :-1].values
y = veri.iloc[:, -1].values

uzunluk=len(X)
X=np.append(np.ones((uzunluk,1)).astype(float), values=X, axis=1)
# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

# Lojistik Regresyon modeli
model = LogisticRegression()  #model=sınıflandırıcı
model.fit(X_train, y_train)  # Düzeltildi: X_train -> y_train


# Tahminler
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)  # Düzeltildi: predict_probe -> predict_proba

# Model değerlendirme
print("Karışıklık Matrisi:")
print(confusion_matrix(y_test, y_pred))
print("/n Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred))

#%%
# Hata matrisinin gorsellestirilmesi
import seaborn as sns
#hata matrisi
hm=confusion_matrix(y_test,y_pred)
print(hm)
print(classification_report(y_test,y_pred))
hm_df=pd.DataFrame(hm,index=["başarısız","başarılı"],columns=["başarısız","başarılı"])
plt.figure(figsize=(8,8))
sns.heatmap(hm_df,annot=True,fmt="g",cmap="Greens")
plt.title("Hata Matrisi",fontsize=16)
plt.ylabel('Gercek degerler',fontsize=16)
plt.xlabel('Tahmin Edilen Degerler', fontsize=16)
plt.show()

#%%
#Yeni gazlem icin LR tahmin sonuçları:
yeni=np.array([[1,2,30,18,10,1,300,15]])
print(model.predict(yeni))
print(model.predict_proba(yeni))