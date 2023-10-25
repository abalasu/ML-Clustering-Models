from turtle import distance
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import pandas as pd

data = pd.read_csv("d:/pythondata/agglomerative.csv")

hierarchical_cluster = AgglomerativeClustering(n_clusters=3, metric='manhattan', linkage='complete')
labels = hierarchical_cluster.fit_predict(data)
print(labels)
plt.scatter(data["Years of Exp"],data["Salary"], c=labels)
plt.show()

linkage_data = linkage(data, method='complete', metric='cityblock', optimal_ordering=True)
dendrogram(linkage_data)
plt.show()