#knn örnek2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Veri setini yükleme
veriseti = pd.read_csv('veriler/ObesityDataSet_raw_and_data_sinthetic.csv')
print(veriseti.head(15))
#%%
# Bağımsız ve bağımlı değişkenleri ayırma
X = veriseti.iloc[:, :-1].values
y = veriseti.iloc[:, -1].values

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
#%%
# Kategorik sütunları sayısallaştırma
le = LabelEncoder()
columns_to_encode = [0, 4, 5, 8, 9, 11, 14, 15]

for i in columns_to_encode:
    X_train[:, i] = le.fit_transform(X_train[:, i])
    X_test[:, i] = le.transform(X_test[:, i])  # fit_transform yerine transform kullanıyoruz

#%%
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
x_train = scaler.fit_transform(X_train)
x_test = scaler.transform(X_test)
#%%
from sklearn.neighbors import KNeighborsClassifier
siniflandirici=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
siniflandirici.fit(x_train, np.ravel(y_train))
y_tahmin=siniflandirici.predict(x_test)
y_olasilik=siniflandirici.predict_proba(x_test)
#%%
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

hm=confusion_matrix(y_test, y_tahmin)
print(hm)
print(classification_report(y_test, y_tahmin))
#%%
# Hata oranı grafiği (1-30 K değerleri)
#hatalarlistesi: Farklı K (komşu sayısı) değerleri için hata oranlarını saklayacak boş bir liste.
hatalarlistesi = []
for k in range(1, 31):
    siniflandirici_k = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2)
    siniflandirici_k.fit(X_train, np.ravel(y_train))
    tahmin_k = siniflandirici_k.predict(X_test)
    hatalarlistesi.append(np.mean(tahmin_k != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,31), hatalarlistesi, 'b.', linestyle='--', markersize=6)
plt.title('1.....30 Aralığındaki K Değerlerine Karşılık Hata Oranları')
plt.xlabel('K Değeri', fontsize=15)
plt.ylabel('Hata Oranı', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid()
plt.show()

#%%
# Yeni veri tahmini (İlk dosyadan düzenlendi)
yeni = np.array([1, 26, 11, 6, 1, 50, 9,1,4,67,12,13,1,45,6,2]).reshape(1, -1)
yeni_scaled = scaler.transform(yeni)
tahmin = siniflandirici.predict(yeni_scaled)
olasilik = siniflandirici.predict_proba(yeni_scaled)
print(f"\nTahmin: {tahmin}, Olasılık: {olasilik}")


