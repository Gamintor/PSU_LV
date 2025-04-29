from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# MNIST podatkovni skup
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train_s = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test_s = x_test.reshape(-1, 28, 28, 1) / 255.0

y_train_s = to_categorical(y_train, num_classes=10)
y_test_s = to_categorical(y_test, num_classes=10)

# 1) Struktura konvolucijske neuronske mreže
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.summary()

# 2) Definiranje karakteristika procesa učenja
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 3) Definiranje callbackova
my_callbacks = [
    callbacks.EarlyStopping(monitor='val_loss', patience=3),
    callbacks.ModelCheckpoint(
        'best_model.h5', monitor='val_accuracy', save_best_only=True)
]

# 4) Provedba treniranja mreže
history = model.fit(x_train_s, y_train_s,
                    epochs=15,
                    batch_size=128,
                    validation_split=0.2,
                    callbacks=my_callbacks)

# 5) Učitavanje najboljeg modela
best_model = keras.models.load_model('best_model.h5')

# 6) Evaluacija mreže
train_loss, train_acc = best_model.evaluate(x_train_s, y_train_s, verbose=0)
test_loss, test_acc = best_model.evaluate(x_test_s, y_test_s, verbose=0)

print(f"\nTočnost na skupu za učenje: {train_acc:.4f}")
print(f"Točnost na skupu za testiranje: {test_acc:.4f}")

# 7) Matrica zabune


def plot_confusion_matrix(y_true, y_pred, classes, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predviđeno')
    plt.ylabel('Stvarno')
    plt.show()


# Predikcije za testni skup
y_pred = best_model.predict(x_test_s)
y_pred_classes = np.argmax(y_pred, axis=1)

plot_confusion_matrix(y_test, y_pred_classes, range(10),
                      'Matrica zabune na testnom skupu')

# Prikaz krivulja učenja
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Točnost treninga')
plt.plot(history.history['val_accuracy'], label='Točnost validacije')
plt.title('Krivulje točnosti')
plt.xlabel('Epoha')
plt.ylabel('Točnost')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Gubitak treninga')
plt.plot(history.history['val_loss'], label='Gubitak validacije')
plt.title('Krivulje gubitka')
plt.xlabel('Epoha')
plt.ylabel('Gubitak')
plt.legend()

plt.tight_layout()
plt.show()
