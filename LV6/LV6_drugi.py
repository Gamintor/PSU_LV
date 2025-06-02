'''cikit-learn kmeans metoda vraća i vrijednost kriterijske funkcije (6-3). Za broj klastera od 1 do 20 odredite vrijednost 
kriterijske funkcije za podatke iz Zadatka 1. Prikažite dobivene vrijednosti pri čemu je na x-osi broj klastera (npr. od 2 
do 20), a na y-osi vrijednost kriterijske funkcije. Kako komentirate dobivene rezultate? Kako biste pomoću dobivenog 
grafa odredili optimalni broj klastera?'''


from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


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


n_samples = 500
flagc = 5
X, y = generate_data(n_samples, flagc)

inertia_values = []
silhouette_scores = []
cluster_range = range(2, 21)
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)


plt.figure(figsize=(10, 7))
plt.plot(cluster_range, inertia_values, marker='x')
plt.title('Vrijednost kriterijske funkcije za različit broj klastera')
plt.xlabel('Broj klastera')
plt.ylabel('Kriterijska vrijednost')
plt.grid()
plt.show()
