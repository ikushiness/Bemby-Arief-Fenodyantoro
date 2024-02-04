# Bemby-Arief-Fenodyantoro
# Load data (asumsikan data tersedia dalam file 'customer_data.csv')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Drop CustomerID (tidak relevan untuk clustering)
data = data.drop('CustomerID', axis=1)

# Standarisasi fitur
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Finding optimal number of clusters using the Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

# Choose the optimal number of clusters based on the Elbow Method
optimal_clusters = 3  # Adjust as needed

# Fit K-Means model
kmeans_model = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
data['Cluster'] = kmeans_model.fit_predict(scaled_features)

# Evaluate clustering performance using Silhouette Score
silhouette_avg = silhouette_score(scaled_features, data['Cluster'])

# Visualize clusters
import seaborn as sns
sns.scatterplot(data['Annual Income (k$)'], data['Spending Score (1-100)'], hue=data['Cluster'], palette='viridis')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Clustering Result')
plt.show()

# Recommendations
print(f"Based on the clustering results, here are some recommendations:\n")
for cluster_num in range(optimal_clusters):
    cluster_data = data[data['Cluster'] == cluster_num]
    print(f"Cluster {cluster_num} - Target audience with specific marketing strategies:")
    print(cluster_data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].describe().iloc[:, 1:])
    print("\n")


# Visualize clusters
import seaborn as sns
sns.scatterplot(data['Annual Income (k$)'], data['Spending Score (1-100)'], hue=data['Cluster'], palette='viridis')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Clustering Result')
plt.show()

# Recommendations
print(f"Based on the clustering results, here are some recommendations:\n")
for cluster_num in range(optimal_clusters):
    cluster_data = data[data['Cluster'] == cluster_num]
    print(f"Cluster {cluster_num} - Target audience with specific marketing strategies:")
    print(cluster_data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].describe().iloc[:, 1:])
    print("\n")
