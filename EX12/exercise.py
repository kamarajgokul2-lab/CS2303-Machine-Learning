import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import dendrogram, linkage


# Load dataset
data = pd.read_csv("product_hierarchical_clustering.csv")

print("Dataset:")
print(data.head())


# Select numerical features
X = data[["Sales", "CustomerRating", "Price", "Stock"]]


# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# -----------------------------------
# DENDROGRAM
# -----------------------------------

linked = linkage(X_scaled, method="ward")

plt.figure(figsize=(10, 5))

dendrogram(
    linked,
    labels=data["ProductName"].values
)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Products")
plt.ylabel("Euclidean Distance")
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()


# -----------------------------------
# HIERARCHICAL CLUSTERING
# -----------------------------------

model = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)

data["Cluster"] = model.fit_predict(X_scaled)


print("\nClustered Products:")

print(
    data[
        [
            "ProductName",
            "Sales",
            "CustomerRating",
            "Price",
            "Stock",
            "Cluster"
        ]
    ]
)


# -----------------------------------
# SILHOUETTE SCORE
# -----------------------------------

score = silhouette_score(
    X_scaled,
    data["Cluster"]
)

print("\nSilhouette Score:", score)


# -----------------------------------
# CLUSTER VISUALIZATION
# -----------------------------------

plt.scatter(
    data["Sales"],
    data["Price"],
    c=data["Cluster"]
)

plt.xlabel("Product Sales")
plt.ylabel("Product Price")

plt.title(
    "Product Groups using Hierarchical Clustering"
)

plt.show()
