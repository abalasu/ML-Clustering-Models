from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import sys
# Centroid - is the center point for the cluster
item_no = [1,2,3,4,5,6,7,8]
prod_type = [12,23,14,23,15,11,20,13]
prod_sub_type = [2,2,3,1,1,2,4,5]
# 
data = list(zip(prod_type, prod_sub_type))
print(data)

# Min Max Scaling - Normalization
MMscaler = MinMaxScaler()
MMscaler.fit(data)
MMscaled_data = MMscaler.transform(data)
print('Normalized Data')
print(MMscaled_data)

# Standard Scaling - Standardization
SSscaler = StandardScaler()
SSscaler.fit(data)
SSscaled_data = SSscaler.transform(data)
print('Standardized Data')
print(SSscaled_data)

# Finding the best possible K value using the Elblow curve
inertia = []
for x in [1, 2, 3, 4]:
    k_means_data_model = KMeans(n_clusters=x, init='k-means++', n_init='auto')
    k_means_data_model.fit(data)
    inertia.append(k_means_data_model.inertia_)
# print(k_means_data_model.cluster_centers_)
# print(k_means_data_model.fit_transform(scaled_data))
#   print('Product Type ', prod_type)
#   print('Sub Type     ', prod_sub_type)
#   print('Cluster      ', k_means_data_model.labels_)
#   plt.scatter(prod_type,prod_sub_type,c=k_means_data_model.labels_)
#   plt.show()
print("Inertia", inertia)
plt.plot(range(1,5), inertia, marker='o')
plt.title("Elbow Curve ")
plt.xlabel("Number of Clusters ")
plt.ylabel("Inertia ")
plt.show()

