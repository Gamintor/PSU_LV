'''Primijenite scikit-learn kmeans metodu za kvantizaciju boje na slici example.png koja dolazi kao prilog ovoj vježbi. 
Prikažite originalnu i kvantiziranu sliku.'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.io import imread, imsave

image = imread("example.png")
original_shape = image.shape

pixels = image.reshape(-1, 3)

kmeans = KMeans(n_clusters=12, random_state=42)
kmeans.fit(pixels)
quantized_pixels = kmeans.cluster_centers_[kmeans.labels_]

quantized_image = quantized_pixels.reshape(original_shape).astype(np.uint8)

imsave("quantized_example.png", quantized_image)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Quantized Image')
plt.imshow(quantized_image)
plt.axis('off')

plt.show()
