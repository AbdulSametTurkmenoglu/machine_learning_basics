from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
#öznitelik seçimi
iris = datasets.load_iris()
model = LogisticRegression(max_iter=1000)
rfe = RFE(model, n_features_to_select=1)
rfe.fit(iris.data, iris.target)
print(iris.feature_names)
print(rfe.support_)
print(rfe.ranking_)

#%%
#öznitelik seçimi
import numpy as np
from sklearn.feature_selection import chi2

# X, her satırı bir örnek (gözlem) ve her sütunu bir özellik (öznitelik) olan 6x3 boyutlu bir matris.
# y, her gözlemin ait olduğu sınıf etiketi.


X = np.array([[1,1,3]
             ,[0,1,5],
              [5,4,1],
              [6,6,2],
              [1,4,0],
              [0,0,0]])
y = np.array([1,1,0,0,2,2])

chi2_stats,p_values = chi2(X,y)
# chi2(X, y), her özelliğin (X'teki sütunların) bağımlı değişken (y) ile olan ilişkisini ölçer.
# Sonuç olarak:
# chi2_stats: Ki-kare istatistik değerlerini verir.
# p_values: Her özelliğin anlamlı olup olmadığını belirten p-değerlerini döndürür.

print(chi2_stats)
print(p_values)
for i in range(len(p_values)):
    if p_values[i] < 0.05:
        print(f"{X[:,i]}.sütun anlamlıdır yani {i}.sütundur")

# Ki-kare değerleri (chi2_stats): Büyük olması, o özelliğin hedef değişkenle daha güçlü ilişkili olduğunu gösterir.
# p-değerleri (p_values): Küçükse (< 0.05), özelliğin istatistiksel olarak anlamlı olduğu kabul edilir.
# Bu sonuçlara göre:
#
# 2. sütunun p-değeri 0.0628, 3. sütunun ise 0.0761. Bunlar 0.05’in biraz üstünde olduğu için istatistiksel olarak sınırda anlamlı diyebiliriz.
# 1.sütunun p-değeri 0.1183, bu yüzden bu sütun hedef değişkenle daha zayıf bir ilişkiye sahip olabilir.

#%%

from sklearn.datasets import load_iris
from sklearn.feature_selection import chi2
#öznitelik seçimi
X,y = load_iris(return_X_y=True)
chi2_stats,p_values = chi2(X,y)
print(chi2_stats)
print(p_values)

#%%

from sklearn.datasets import load_iris
from sklearn.feature_selection import f_classif

X,y = load_iris(return_X_y=True)
f_statistics,p_values = f_classif(X,y)
#f_classif: ANOVA F-test (Analysis of Variance F-test) istatistiğini hesaplamak için kullanılır.
# Bu test, her özniteliğin bağımlı değişkenle (sınıf etiketiyle) ilişkisini ölçer.
print(f_statistics)
print(p_values)

#%%

from sklearn.datasets import make_classification
from sklearn.feature_selection import f_classif

X,y = make_classification(n_samples=100,n_features=10,n_informative=2,
                          n_clusters_per_class=1,shuffle=False,
                          random_state=42)

#n_samples=100 → 100 satırdan oluşan bir veri kümesi oluşturur.
#n_informative=2 → Sadece 2 tanesi hedef değişkenle gerçekten ilişkili olacak. (Diğer 8 öznitelik rastgele gürültü (noise) içerecek.)
#n_clusters_per_class=1 → Her sınıf için 1 küme oluşturur.
#shuffle=False → Veriler karıştırılmaz.
#random_state=42 → Rastgelelik kontrol edilir (sonuçlar tekrar üretilebilir olur).
f_statistics,p_values = f_classif(X,y)
print(f_statistics)
print(p_values)

#%%

import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

Immunotherapy = pd.read_excel("veriler/Immunotherapy.xlsx")
model = LogisticRegression()
rfe = RFE(model, n_features_to_select=4)
rfe.fit(Immunotherapy.iloc[:, 0:6], Immunotherapy.iloc[:, 7])
# Immunotherapy.iloc[:, 0:6] → İlk 6 sütun girdi (X) olarak seçilmiş. (0’dan 5. sütuna kadar)
# Immunotherapy.iloc[:, 7] → 7. sütun çıktı (y) olarak seçilmiş.
print(rfe.support_)
print(rfe.ranking_)

#%%

import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

Immunotherapy = pd.read_excel("veriler/Immunotherapy.xlsx")
model = SVR(kernel='linear')
rfe = RFE(model, n_features_to_select=4)
rfe.fit(Immunotherapy.iloc[:, 0:6], Immunotherapy.iloc[:, 7])

print(rfe.support_)
print(rfe.ranking_)

#%%

import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
canak_yaprak_uzunlugu = iris.data[:,0]

iris = datasets.load_iris()
canak_yaprak_uzunlugu = iris.data[:,0]
ort = np.mean(canak_yaprak_uzunlugu)
s_sapma = np.std(canak_yaprak_uzunlugu)

X = np.arange(150,dtype=float)

for i in range(150):
    X[i] = (canak_yaprak_uzunlugu[i]-ort)/s_sapma
print(X)

#%%

import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
canak_yaprak_uzunlugu = iris.data[:,0]

min = np.min(canak_yaprak_uzunlugu)
max = np.max(canak_yaprak_uzunlugu)

X = np.arange(150,dtype=float)

for i in range(150):
    X[i] = (canak_yaprak_uzunlugu[i]-min)/(max-min)
print(X)

#%%

import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np

veriseti = pd.read_excel("veriler/VeriOnIsleme.xlsx")
X = veriseti.iloc[:,:-1].values
y = veriseti.iloc[:,-1].values
#Bağımsız değişkenler (X) → Veri setinin son sütunu hariç tüm sütunları içerir.
#Bağımlı değişken (y) → Son sütunu içerir (Hedef değişken).

yaklasik_deger = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
#eksik verileri doldurmak için
X[:,1:5] = yaklasik_deger.fit_transform(X[:,1:5])
print(yaklasik_deger.statistics_)

#%%

import pandas as pd

veriseti = pd.read_excel("veriler/VeriOnIsleme.xlsx")
X = veriseti.iloc[:,:-1].values
y = veriseti.iloc[:,-1].values
X[:,0:5] = yaklasik_deger.fit_transform(X[:,0:5])
print(yaklasik_deger.statistics_)

#%%
#benim denediğim
import matplotlib.pyplot as plt
import numpy as np

plt.axis('auto')
y=np.array([2,3,4,5,7,9,11,14,15])
x=np.arange(len(y))
plt.plot(x,y,color='black')
esik=9
altesik=y<esik
plt.scatter(x[altesik],y[altesik],color='black')
ustesik=np.logical_not(altesik)
plt.scatter(x[ustesik],y[ustesik],color='red')
plt.axhline(esik,color='red',linestyle='--')

plt.show()