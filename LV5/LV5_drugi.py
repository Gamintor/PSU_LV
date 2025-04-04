''' 
 Izgradite i evaluirajte algoritam K najbližih susjeda. Slijedite ovaj redoslijed: 
 
a) Podijelite  podatke  na  skup  za  učenje  i  skup  za  testiranje  (omjer  80%-20%)  pomoću  funkcije 
train_test_split. Koristite opciju stratify=y.

b) Pomoću  StandardScaler skalirajte ulazne veličine. 

c) Pomoću klase KNeighborsClassifier izgradite algoritam K najbližih susjeda. 

d) Evaluirajte izgrađeni klasifikator na testnom skupu podataka: 

    a. prikažite matricu zabune 
    b. izračunajte točnost klasifikacije 
    c. izračunajte preciznost i odziv po klasama 
e) Što se događa s rezultatima ako se koristi veći odnosno manji broj susjeda? 
    - mijenjamo izgled modela, ako stavimo premali k moze biti prejednostavan nacina klasificiranja, 
    a ako stavimo prevelik k moze bit overtrained sto isto ne valja 

f) Što se događa s rezultatima ako ne koristite skaliranje ulaznih veličina?
    - moze doci do poremecaja pri prikazu ili odlucivanju tokom klasifikacije 
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, accuracy_score, classification_report

df = pd.read_csv("occupancy_processed.csv")

feature_names = ['S3_Temp', 'S5_CO2']
target_name = 'Room_Occupancy_Count'

x = df[feature_names].to_numpy()
y = df[target_name].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Matrica zabune:\n", conf_matrix)

accuracy = accuracy_score(y_test, y_pred)
print("Tocnost modela:", accuracy)

print("Izvjestaj klasifikacije:\n", classification_report(
    y_test, y_pred, target_names=['Slobodna', 'Zauzeta']))

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])

disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Izracunaj preciznost
precision = precision_score(y_test, y_pred)

# Izracunaj tocnost
accuracy = accuracy_score(y_test, y_pred)

# Izracunaj odziv
recall = recall_score(y_test, y_pred)

# Report
print(classification_report(y_test, y_pred))
