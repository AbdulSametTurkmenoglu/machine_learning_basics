import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split

veriseti = pd.read_excel('veriler/Immunotherapy1.xlsx')

X = veriseti.iloc[:, :-1].values
y = veriseti.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

from sklearn.preprocessing import MinMaxScaler
#veri ayırdıktan sonra min max yapılır veri sızıntısı olmasın diye
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
#testte fit yapmaya gerek yok.Min maxı olmasın diye.Çünkü trainin min max lazım
X_test = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
siniflandirici = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
#n_neighbors=5: 5 en yakın komşu kullanılarak sınıflandırma yapılır.
#metric='minkowski' ve p=2: Öklid (Euclidean) mesafesi kullanılır
# (p=2, Minkowski mesafe metriğinde Öklid mesafesine denk gelir).
#y_train.ravel(): Eğer y_train tek sütunlu bir vektörse, .ravel() kullanarak 1D array haline getirilir.
siniflandirici.fit(X_train, y_train.ravel())

y_tahmin = siniflandirici.predict(X_test)
y_olasilik = siniflandirici.predict_proba(X_test)

#%%
#hata matrisi
hm=confusion_matrix(y_test,y_tahmin)
print(hm)
print(classification_report(y_test,y_tahmin))
#%%
# Hata matrisinin gorsellestirilmesi
import seaborn as sns

hm_df=pd.DataFrame(hm,index=["başarısız","başarılı"],columns=["başarısız","başarılı"])
plt.figure(figsize=(8,8))
sns.heatmap(hm_df,annot=True,fmt="g",cmap="Greens")
plt.title("Hata Matrisi",fontsize=16)
plt.ylabel('Gercek degerler',fontsize=16)
plt.xlabel('Tahmin Edilen Degerler', fontsize=16)
plt.show()

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
# ROC Eğrisi (İlk dosyadan düzenlendi)
#Yanlış Pozitif Oran (False Positive Rate - FPR) ve Doğru Pozitif Oran
# (True Positive Rate - TPR) değerlerini kullanarak modelin farklı eşik değerlerindeki performansını gösterir.
#FPR (Yanlış Pozitif Oran - YPO): Modelin yanlış pozitif tahmin yapma oranı.
#TPR (Doğru Pozitif Oran - DPO): Modelin doğru pozitif tahmin yapma oranı.
#AUC (Area Under Curve - Eğri Altındaki Alan):
#AUC değeri 1’e ne kadar yakınsa, model o kadar iyi sınıflandırma yapıyor demektir.
#0.5’e yakınsa model rastgele tahmin yapıyordur.
#1.0’a yakınsa model mükemmel sınıflandırma yapıyordur.
#Gerçek etiketler (y_test) ile modelin tahmin ettiği etiketler (y_tahmin) kullanılarak FPR (Yanlış Pozitif Oran), TPR (Doğru Pozitif Oran) ve eşik değerleri hesaplanır.
#fpr: Yanlış pozitif oranlar (X ekseni).
#tpr: Doğru pozitif oranlar (Y ekseni).
fpr, tpr, thresholds = roc_curve(y_test, y_tahmin)
auc_degeri = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label='AUC %0.2f' % auc_degeri)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Yanlış Pozitif Oran (YPO)', fontsize=15)
plt.ylabel('Doğru Pozitif Oran (DPO)', fontsize=15)
plt.title('ROC Eğrisi')
plt.legend(loc="best")
plt.grid()
plt.show()

# Yeni veri tahmini (İlk dosyadan düzenlendi)
yeni = np.array([2, 26, 11, 6, 1, 50, 9]).reshape(1, -1)
yeni_scaled = scaler.transform(yeni)
tahmin = siniflandirici.predict(yeni_scaled)
olasilik = siniflandirici.predict_proba(yeni_scaled)
print(f"\nTahmin: {tahmin}, Olasılık: {olasilik}")