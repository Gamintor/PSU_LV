import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("tiger.png")
img = img[:, :, 0].copy()

print(img.shape)
print(img.dtype)

plt.figure()

plt.imshow(img*0.16, cmap="gray", vmax=100)
plt.show()
plt.imshow(np.rot90(img), cmap="gray")
plt.show()
plt.imshow(np.fliplr(img), cmap="gray")
plt.show()
