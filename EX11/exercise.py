import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load dataset
data = pd.read_csv("retail_customer_clustering.csv")

print("Dataset:")
print(data.head())

# Select features for clustering
X = data[["AnnualIncome", "SpendingScore"]]

# -------------------------------
# Elbow Method
# -------------------------------
wcss = []

for k in range(1, 7):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X)
    wcss.append(model.inertia_)

plt.plot(range(1, 7), wcss, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

# -------------------------------
# K-Means Clustering
# -------------------------------
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

data["Cluster"] = kmeans.fit_predict(X)

print("\nClustered Data:")
print(data[["CustomerID", "AnnualIncome",
            "SpendingScore", "Cluster"]])

# -------------------------------
# Silhouette Score
# -------------------------------
score = silhouette_score(X, data["Cluster"])

print("\nSilhouette Score:", score)

# -------------------------------
# Cluster Centers
# -------------------------------
print("\nCluster Centers:")
print(kmeans.cluster_centers_)

# -------------------------------
# Cluster Visualization
# -------------------------------
plt.scatter(
    data["AnnualIncome"],
    data["SpendingScore"],
    c=data["Cluster"]
)

# Plot centroids
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker="X",
    s=200
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation using K-Means")
plt.show()
