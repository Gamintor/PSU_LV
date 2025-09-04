
'''Potrebno je izgraditi potpuno povezanu unaprijednu višeslojnu neuronsku mrežu na temelju MNIST skupa podataka te 
izvršiti njenu evaluaciju.  U prilogu vježbe nalazi se skripta 7.1. koja učitava MNIST skup podataka. Dopunite skriptu na 
odgovarajućim mjestima: 
 
1) Prikažite nekoliko slika iz MNIST skupa podataka (pomoću matplotlib biblioteke). 
 
2) Izgradite potpuno povezanu neuronsku mrežu pomoću Keras API. Mreža treba imati strukturu danu slikom 7.1. 
https://keras.io/guides/sequential_model/ 
 
3) Izračunajte točnost izgrađene mreže na skupu podataka za učenje i skupu podataka za testiranje. 
 
4) Prikažite matricu zabune na skupu podataka za učenje i na skupu podataka za testiranje. Komentirajte dobivene 
rezultate. '''

'''Zadatak 2 
 
Prikažite nekoliko nasumičnih primjera iz testnog skupa podataka koje je izgrađena mreža pogrešno klasificirala (napišite 
iznad slike koja je stvarna oznaka slike i oznaku koju je mreža procijenila).'''



import numpy as np  
from tensorflow import keras
from tensorflow.keras import layers  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

for i in range(5,10):
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Broj: {y_train[i]}")
    plt.show()

x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

x_train_s = x_train_s.reshape(60000, 784)
x_test_s = x_test_s.reshape(10000, 784)

y_train_s = keras.utils.to_categorical(y_train, 10)
y_test_s = keras.utils.to_categorical(y_test, 10)

model = Sequential()
model.add(keras.Input(shape=(784,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

fit = model.fit(x_train_s, y_train_s, epochs=5, batch_size=32)

train_loss, train_accuracy = model.evaluate(x_train_s, y_train_s)
test_loss, test_accuracy = model.evaluate(x_test_s, y_test_s)
print(f"Trening Tocnost: {train_accuracy:.2f}")
print(f"Test Tocnost: {test_accuracy:.2f}")

y_test_pred = model.predict(x_test_s)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)
conf_matrix = confusion_matrix(y_test, y_test_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=range(10))
disp.plot()
plt.show()

netocno = np.where(y_test != y_test_pred_classes)[0]
for i in range(5):
    index = netocno[i]
    plt.imshow(x_test[index], cmap='gray')
    plt.title(f"Istinito: {y_test[index]}, Predvideno: {y_test_pred_classes[index]}")
    plt.show()
