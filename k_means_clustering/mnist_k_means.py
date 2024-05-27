import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

digits = datasets.load_digits()
# print(digits.DESCR)
# print(digits.target)

plt.gray()
plt.matshow(digits.images[100])
plt.show()
plt.clf()

print(digits.target[100])

km = KMeans(n_clusters = 10, random_state = 42)
km.fit(digits.data)

fig = plt.figure(figsize = (8, 3))

fig.suptitle("Cluser Center Images", fontsize = 14, fontweight = "bold")

for i in range(10):
  ax = fig.add_subplot(2, 5, 1 + i)

  ax.imshow(km.cluster_centers_[i].reshape((8, 8)), cmap = plt.cm.binary)

plt.show()

new_samples = np.array([
[0.00,0.00,0.00,0.46,0.38,0.00,0.00,0.00,0.00,4.42,7.24,7.62,7.17,2.14,0.00,0.00,2.36,7.62,4.73,3.05,6.40,6.56,0.00,0.00,1.68,5.64,0.31,0.23,6.63,5.79,0.00,0.00,0.00,0.00,1.14,6.03,7.55,1.98,0.00,0.00,0.00,2.06,7.32,7.17,2.36,0.00,0.99,0.31,0.00,4.58,7.62,7.62,7.62,7.62,7.62,4.19,0.00,0.15,2.44,3.05,3.05,3.05,3.05,0.69],
[0.00,0.00,0.00,0.31,0.53,0.00,0.00,0.00,0.00,1.45,5.64,7.55,7.55,1.76,0.00,0.00,0.76,7.40,6.94,3.66,7.24,4.42,0.00,0.00,0.15,3.51,0.76,0.15,6.41,6.02,0.00,0.00,0.00,0.00,3.13,6.79,7.62,2.82,0.00,0.00,0.00,0.92,7.62,7.62,6.56,3.89,1.75,0.00,0.00,0.00,2.90,5.87,6.56,6.86,3.43,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
[0.00,0.08,1.45,2.29,2.14,0.23,0.00,0.00,0.00,6.56,7.62,7.62,7.62,6.33,0.08,0.00,0.00,4.12,2.59,0.99,3.13,7.62,1.37,0.00,0.00,0.00,0.00,0.00,2.36,7.62,1.45,0.00,0.00,0.00,0.15,2.44,6.86,6.48,0.15,0.00,0.99,5.34,7.62,7.62,7.24,2.21,0.61,0.00,2.29,7.55,7.62,7.62,7.62,7.62,7.55,0.53,0.00,0.38,1.45,1.52,1.68,2.29,1.98,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.76,5.95,6.10,6.79,6.86,5.34,0.38,0.00,0.46,4.42,4.57,4.42,4.50,7.62,2.67,0.00,0.00,0.00,0.00,0.00,2.13,7.62,1.83,0.00,0.00,0.00,0.00,2.14,6.48,6.71,0.15,0.00,0.00,0.92,6.79,7.62,6.71,1.60,0.00,0.00,0.00,1.75,7.62,7.40,4.80,4.57,2.44,0.00,0.00,0.00,1.68,5.57,6.10,6.79,7.40,0.23]
])
new_labels = km.predict(new_samples)

for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(0, end='')
  elif new_labels[i] == 1:
    print(9, end='')
  elif new_labels[i] == 2:
    print(2, end='')
  elif new_labels[i] == 3:
    print(1, end='')
  elif new_labels[i] == 4:
    print(6, end='')
  elif new_labels[i] == 5:
    print(2, end='')
  elif new_labels[i] == 6:
    print(4, end='')
  elif new_labels[i] == 7:
    print(5, end='')
  elif new_labels[i] == 8:
    print(7, end='')
  elif new_labels[i] == 9:
    print(3, end='')