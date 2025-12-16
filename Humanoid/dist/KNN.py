import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

k = 10
n = 100
num_of_tst_pts = 5

x1, y1 = np.random.multivariate_normal([3, 10], [[5, 0], [0, 6]], n).T
x2, y2 = np.random.multivariate_normal([10, 4], [[6, 0], [0, 5]], n).T
dog = []
cat = []
i = 0
while (i < n):
    dog.append([x1[i], y1[i]])
    cat.append([x2[i], y2[i]])
    i += 1
given_data = {'dog': dog, 'cat': cat}
find = np.random.randint(0, 20, (num_of_tst_pts, 2))

plt.subplot(1, 2, 1)
for i in find:
    plt.scatter(i[0], i[1], s=69, color='k')
plt.scatter(x1, y1, s=10, color='r')
plt.scatter(x2, y2, s=10, color='b')

def KNN_predict(given_data, predict, k):
    distance_list = []
    for animal in given_data:
        for point in given_data[animal]:
            distance = np.linalg.norm(np.array(point) - np.array(predict))
            distance_list.append([distance, animal])
    sorted_dist = [i[1] for i in sorted(distance_list)[:k]]
    result = Counter(sorted_dist).most_common(1)[0][0]
    is_dog = (result == 'dog')
    return is_dog

plt.subplot(1, 2, 2)
plt.scatter(x1, y1, s=10, color=(1, 0.4, 0.4))
plt.scatter(x2, y2, s=10, color=(0.4, 0.4, 1))
for i in find:
    if (KNN_predict(given_data, i, k)):
        print("Dog", )
        plt.scatter(i[0], i[1], s=69, color='r')
    else:
        print("Cat")
        plt.scatter(i[0], i[1], s=69, color='b')
plt.show()

input("press enter to exit")