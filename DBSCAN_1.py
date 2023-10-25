from sklearn.cluster import DBSCAN
import pandas as pd
from matplotlib import pyplot as plt

color_dict = {-1:'red',0:'black',1:'green',2:'orange',3:'grey',4:'yellow',5:'blue',6:'magenta',7:'cyan'}
data = pd.read_csv('d:/pythondata/mall_customers.csv')
X_train = data[['Age','Annual Income (k$)', 'Spending Score (1-100)']]
X1_train = data[['Annual Income (k$)', 'Spending Score (1-100)']]
X2_train = data[['Age', 'Spending Score (1-100)']]
a = 'Age'
b = 'Annual Income (k$)'
c = 'Spending Score (1-100)'
X_Age = data[a]
X_Annual_Income = data[b]
X_Spending_Score = data[c]
dbmodel = DBSCAN(eps=10,min_samples=5)
dbmodel.fit(X1_train)
print(dbmodel.labels_)
i = 0
while i<len(dbmodel.labels_):
    c = color_dict.get(dbmodel.labels_[i])
    plt.scatter(X_Annual_Income[i],X_Spending_Score[i],c=c)
    i += 1
plt.title('DBSCAN Clustering',loc='center')
plt.xlabel('Annual Income in k$')
plt.ylabel('Spending Score (1-100)')
plt.show()

# With all 3 features

dbmodel = DBSCAN(eps=10,min_samples=3)
dbmodel.fit(X_train)
print(dbmodel.labels_)
i = 0
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
while i<len(dbmodel.labels_):
    c = color_dict.get(dbmodel.labels_[i])
    ax.scatter(X_Age[i], X_Annual_Income[i],X_Spending_Score[i],c=c)
    i += 1
ax.set_title('DBSCAN Clustering',loc='center')
ax.set_xlabel('Age')
ax.set_ylabel('Annual Income in k$')
ax.set_zlabel('Spending Score (1-100)')
plt.show()
