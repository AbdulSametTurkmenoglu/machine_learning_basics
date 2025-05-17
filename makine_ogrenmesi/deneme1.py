import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Verileri yükle
veriler = pd.read_csv('satislar.csv')

# Verileri hedef değişken ve özellikler olarak ayır
X = veriler.iloc[:, :-1]
y = veriler.iloc[:, -1]

# Önceki hali
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=y)
plt.title('Önceki Hali')

# Min-max normalizasyonu uygula
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Sonraki hali
plt.subplot(1, 2, 2)
sns.scatterplot(x=X_normalized[:, 0], y=X_normalized[:, 1], hue=y)
plt.title('Sonraki Hali (Min-Max Normalizasyonu)')

plt.show()

# Normalizasyon sonrası verileri eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# KNN modelini oluştur ve eğit
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Modeli test et
y_pred = knn.predict(X_test)

# Doğruluk skoru
accuracy = accuracy_score(y_test, y_pred)
print(f'Doğruluk: {accuracy}')

# F1 skoru
f1 = f1_score(y_test, y_pred, average='weighted')
print(f'F1 Skoru: {f1}')

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(cm)

# Test verileri ve tahminleri birleştir
results = pd.DataFrame({'x': X_test[:, 0], 'y': X_test[:, 1], 'actual': y_test, 'predicted': y_pred})

# Sonuçları görselleştir
plt.figure(figsize=(10, 6))
sns.scatterplot(x='x', y='y', hue='predicted', palette='deep', data=results)
plt.title('KNN ile Sınıflandırma Sonuçları')
plt.show()
