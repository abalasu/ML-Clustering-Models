import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

# Finding the best value for K using the Elbow method
inertias = []
# inertia_ is the measure of the distance between each data point, squaring it and adding it. Smaller inertia means closer points
data = list(zip(x,y))
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,n_init='auto')
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)
print(inertias)
plt.plot(range(1,11), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# Based on the level of steepness of the curve we consider 2 as the best value for K 
i = input("Press any key ")
plt.close()

kmeans = KMeans(n_clusters=2, n_init='auto')
kmeans.fit(data)

plt.scatter(x, y, c=kmeans.labels_)
plt.show()
