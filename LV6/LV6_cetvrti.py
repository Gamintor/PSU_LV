'''Primijenite scikit-learn kmeans metodu za kvantizaciju boje na slici. Proučite kod 6.2. iz priloga vježbe te ga primijenite 
za kvantizaciju boje na slici example_grayscale.png koja dolazi kao prilog ovoj vježbi. Mijenjajte broj klastera. 
Što primjećujete? Izračunajte kolika se kompresija ove slike može postići ako se koristi 10 klastera. 
Pomoću sljedećeg koda možete učitati sliku: 
 
import matplotlib.image as mpimg 
 
imageNew = mpimg.imread('example_grayscale.png') '''

# -*- coding: utf-8 -*-
"""
Created on Sun Dec 02 12:08:00 2018

@author: Grbic
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn import cluster
import numpy as np


face = mpimg.imread('example_grayscale.png')
if len(face.shape) == 3:
    face = np.mean(face, axis=2)

X = face.reshape((-1, 1))
k_means = cluster.KMeans(n_clusters=5, n_init=10)
k_means.fit(X)
values = k_means.cluster_centers_.squeeze()
labels = k_means.labels_
face_compressed = np.choose(labels, values)
face_compressed.shape = face.shape

plt.figure(1)
plt.imshow(face,  cmap='gray')
plt.show()

plt.figure(2)
plt.imshow(face_compressed,  cmap='gray')
plt.show()
