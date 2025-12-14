import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


df = pd.read_excel("Strava Running Data.xlsx")

df = df[df["sport_type"].str.contains("Run", case=False, na=False)].copy()

df["distance_km"] = df["distance"] / 1000           
df["moving_time_min"] = df["moving_time"] / 60      
df["pace_min_per_km"] = df["moving_time_min"] / df["distance_km"]


df = df[df["pace_min_per_km"].between(3, 15)]
df = df[df["distance_km"].between(0.5, 60)]


features = [
    "distance_km",
    "moving_time_min",
    "pace_min_per_km",
    "average_speed",          
    "total_elevation_gain"
]

X = df[features].copy()


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


inertias = []
sil_scores = []
K_RANGE = range(2, 11)

for k in K_RANGE:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    inertias.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

# Elbow plot
plt.figure(figsize=(8, 4))
plt.plot(K_RANGE, inertias, marker="o")
plt.title("Elbow Method for Choosing k")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (Within-cluster SSE)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Silhouette plot
plt.figure(figsize=(8, 4))
plt.plot(K_RANGE, sil_scores, marker="o")
plt.title("Silhouette Scores for Different k")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.tight_layout()
plt.show()


BEST_K = 4   

kmeans = KMeans(n_clusters=BEST_K, n_init=10, random_state=42)
df["cluster"] = kmeans.fit_predict(X_scaled)

cluster_centers = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=features
)
print("\nCluster centers (in original units):")
print(cluster_centers.round(2))

plt.figure(figsize=(8, 5))
for c in range(BEST_K):
    subset = df[df["cluster"] == c]
    plt.scatter(
        subset["distance_km"],
        subset["pace_min_per_km"],
        alpha=0.6,
        label=f"Cluster {c}"
    )

plt.xlabel("Distance (km)")
plt.ylabel("Pace (min/km)")
plt.title("Clusters of Runs: Distance vs Pace")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
for c in range(BEST_K):
    subset = df[df["cluster"] == c]
    plt.scatter(
        subset["average_speed"],
        subset["pace_min_per_km"],
        alpha=0.6,
        label=f"Cluster {c}"
    )

plt.xlabel("Average Speed (m/s)")
plt.ylabel("Pace (min/km)")
plt.title("Clusters of Runs: Speed vs Pace")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

df.to_csv("Strava_with_clusters.csv", index=False)
print("\nSaved clustered data to Strava_with_clusters.csv")
