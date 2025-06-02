'''Primijenite  hijerarhijsko  grupiranje  na  podatke  korištene  u  Zadatku  1  pomoću  funkcije  linkage  koja  je  ugrađena 
scipy  metoda za agglomerative clustering: 
 
from scipy.cluster.hierarchy import dendrogram, linkage 
 
Prikažite pripadni dendogram. Mijenjajte korištenu metodu (argument method). Kako komentirate postignute rezultate?'''


from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage


def generate_data(n_samples, flagc):
    if flagc == 1:
        random_state = 365
        X, y = datasets.make_blobs(
            n_samples=n_samples, random_state=random_state)
    elif flagc == 2:
        random_state = 148
        X, y = datasets.make_blobs(
            n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)
    elif flagc == 3:
        random_state = 148
        X, y = datasets.make_blobs(n_samples=n_samples, centers=4, cluster_std=[
                                   1.0, 2.5, 0.5, 3.0], random_state=random_state)
    elif flagc == 4:
        X, y = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
    elif flagc == 5:
        X, y = datasets.make_moons(n_samples=n_samples, noise=.05)
    else:
        X, y = [], []
    return X, y


n_samples = 25
flagc = 5
X, y = generate_data(n_samples, flagc)

linked = linkage(X, method="ward")

plt.figure(figsize=(10, 7))
dendrogram(linked, labels=[f"Podatak {i+1}" for i in range(len(X))])
dendrogram(linked, labels=[f"Podatak {i+1}" for i in range(len(X))])
plt.ylabel('Udaljenost')
plt.show()
