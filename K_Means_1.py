from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, completeness_score
import numpy as np

x = [2,2,8,5,7,6,1,4,6,8,10,12, 13, 14]
y = [10,5,4,8,5,4,2,9,5,4,3,11, 12, 15]
init_1 = (2,10)
init_2 = (5,8)
init_3 = (1,2)
start_points = [init_1, init_2, init_3]
data = list(zip(x,y))
print(data)
plt.scatter(x,y)
plt.show()
k_means_data_model = KMeans(n_clusters=3, n_init='auto', random_state=1)
k_means_data_model.fit(data)
print(k_means_data_model.cluster_centers_)
# print(k_means_data_model.transform(data))
print(k_means_data_model.labels_)
plt.scatter(x,y,c=k_means_data_model.labels_)
for center in k_means_data_model.cluster_centers_:
    center = center[:2]
    plt.scatter(center[0],center[1],marker = '^',c = 'red')
plt.show()
act_tup = (0, 2, 1, 0, 1, 1, 2, 0, 1, 2, 0, 1, 2, 2)
actual_labels = np.array(act_tup)
print("Silhoette Score")
print(silhouette_score(data, k_means_data_model.labels_, metric='euclidean'))
print("Completeness Score")
print(completeness_score(actual_labels, k_means_data_model.labels_))

