from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

model = keras.models.load_model('model.h5')

image_path = 'test.png'
img = Image.open(image_path).convert('L') 
img = img.resize((28, 28))  
img_array = np.array(img) / 255.0  
img_array = img_array.reshape(1, 28, 28, 1) 


predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)

print(f'Predicted class: {predicted_class}')
plt.imshow(img, cmap='gray')
plt.title(f'Predicted: {predicted_class}')
plt.axis('off')
plt.show()
