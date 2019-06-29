from sklearn.cluster import KMeans
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random

# data

# X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
# maskers = ['.', ',', 'o', 'v', '^', '<', '>', 's', 'p', '1', '*', 'h', '+', 'x', 'd', '|', '_']
maskers = ['.']
colors = ['b', 'g', 'r', 'black', 'brown', 'cyan', 'darkgray', 'dimgray', 'gold', 'lime']
data_list = []
for i in range(1000):
    x = random.randrange(0, 720)
    y = random.randrange(0, 900)
    data_list.append([x, y])

data_array = np.array(data_list)
print(data_array)
kmeans = KMeans(n_clusters=20, random_state=0).fit(data_array)


for i, l in enumerate(kmeans.labels_):
    plt.plot(data_array[i][0], data_array[i][1], color=colors[l % colors.__len__()], marker=maskers[l % maskers.__len__()], ls='None')
plt.show()

a = kmeans.predict([[1, 2]])
print(a[0])
