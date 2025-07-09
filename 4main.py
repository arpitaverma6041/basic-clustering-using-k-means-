
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# reading dataset
data = pd.read_csv('4data.csv')   # dataset will have Age and Income columns
print("First few rows of data:")
print(data.head())

# scaling the data 
scaler = StandardScaler()
scaled = scaler.fit_transform(data)

# applying KMeans clustering (3 clusters)
model = KMeans(n_clusters=3, random_state=42)
clusters = model.fit_predict(scaled)

# adding cluster column to original data
data['Cluster'] = clusters

# plotting the results
plt.figure(figsize=(7,5))
plt.scatter(scaled[:,0], scaled[:,1], c=clusters, cmap='rainbow', edgecolor='black')
plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], 
            s=150, c='red', marker='X', label='Center')
plt.xlabel("Age (scaled)")
plt.ylabel("Income (scaled)")
plt.title("KMeans Clustering - Age vs Income")
plt.legend()
plt.grid()
plt.show()