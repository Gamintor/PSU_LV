'''
Po uzoru na prethodne zadatke izgradite model logističke regresije. Što primjećujete kod vrednovanja ovog modela? Što 
je uzrok dobivenim rezultatima?

   - primjetio sam po matrici zabune i tocnosti da logicka regresija vise grijesi
'''


'''
Umjesto algoritma K najbližih susjeda koristite stablo odlučivanja te ponovite korake a) do d) iz prethodnog zadatka. 
a) Vizualizirajte dobiveno stablo odlučivanja. 
b) Što se događa s rezultatima ako mijenjate parametar max-depth stabla odlučivanja? 
    - smanjujemo ili povecavamo kolicinu grananja, to moze dovesti do overtraniranja
c) Što se događa s rezultatima ako ne koristite skaliranje ulaznih veličina?'''


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

f) Što se događa s rezultatima ako ne koristite skaliranje ulaznih veličina?
'''


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("occupancy_processed.csv")

feature_names = ['S3_Temp', 'S5_CO2']
target_name = 'Room_Occupancy_Count'

X = df[feature_names].to_numpy()
y = df[target_name].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

reg = LogisticRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Matrica zabune:\n", conf_matrix)


accuracy = accuracy_score(y_test, y_pred)
print("Točnost modela:", accuracy)

print("Izvještaj klasifikacije:\n", classification_report(
    y_test, y_pred, target_names=['Slobodna', 'Zauzeta']))


cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,  display_labels=['Class  0',
                                                                    'Class 1'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
