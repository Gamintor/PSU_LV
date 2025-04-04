'''

Room occupancy classification 
Skripta 5.1. učitava skup podataka koji se nalazi u csv datoteci occupancy_processed.csv. Ova datoteka sadrži 
podatke koji su prikupljeni u prostoriji veličine 6m x 4.6m tijekom 4 dana [1]. Zbog jednostavnosti skup sadrži samo dva 
atributa: mjerenja dobivena sa senzora temperature i mjerenja sa senzora CO2. Izlazna (ciljna) veličina je zauzetost 
prostorije  (0  –  prazna  prostorija,  1  –  u  prostoriji  se  nalazi  barem  jedna  osoba).  Cilj je izgraditi klasifikator koji će na 
temelju trenutnih mjerenja dobivenih sa senzora temperature i sa senzora CO2 procijeniti zauzetost prostorije. 
 
a) Pokrenite skriptu i pogledajte dobiveni dijagram raspršenja. Što primjećujete? 
   - možemo na nekim mjestima primjetiti da nije najbolje definirano je li zauzeta ili slobodna prostorija
b) Koliko podatkovnih primjera sadrži učitani skup podataka? 
   - Sadrži 10129 primjera
c) Kakva je razdioba podatkovnih primjera po klasama? 
   - logistička razdioba
R.Grbic, 2024.

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ucitaj podatke za ucenje
df = pd.read_csv('occupancy_processed.csv')
print(len(df))

feature_names = ['S3_Temp', 'S5_CO2']
target_name = 'Room_Occupancy_Count'
class_names = ['Slobodna', 'Zauzeta']

X = df[feature_names].to_numpy()
y = df[target_name].to_numpy()

# Scatter plot
plt.figure()
for class_value in np.unique(y):
    mask = y == class_value
    plt.scatter(X[mask, 0], X[mask, 1], label=class_names[class_value])

plt.xlabel('S3_Temp')
plt.ylabel('S5_CO2')
plt.title('Zauzetost prostorije')
plt.legend()
plt.show()
